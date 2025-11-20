import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.models.llama import LlamaConfig, modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.int_llama_layer import QuantLlamaAttention


def _make_quant_args():
    quant_params = {"quant_method": None}
    return SimpleNamespace(
        k_weight_quant_params=quant_params,
        k_act_quant_params=quant_params,
        v_weight_quant_params=quant_params,
        v_act_quant_params=quant_params,
        q_weight_quant_params=quant_params,
        q_act_quant_params=quant_params,
        o_weight_quant_params=quant_params,
        o_act_quant_params=quant_params,
        q_quant_params=quant_params,
        k_quant_params=quant_params,
        p_quant_params=quant_params,
        v_quant_params=quant_params,
        gate_weight_quant_params=quant_params,
    )


def _reference_attention(attn_module, hidden_states, attention_mask=None):
    bsz, q_len, _ = hidden_states.size()
    query_states = attn_module.q_proj(hidden_states).view(
        bsz, q_len, attn_module.num_heads, attn_module.head_dim
    ).transpose(1, 2)
    key_states = attn_module.k_proj(hidden_states).view(
        bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim
    ).transpose(1, 2)
    value_states = attn_module.v_proj(hidden_states).view(
        bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = attn_module.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

    key_states = repeat_kv(key_states, attn_module.num_key_value_groups)
    value_states = repeat_kv(value_states, attn_module.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_module.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sliding_window = getattr(attn_module.config, "sliding_window", None)
    if sliding_window is not None and sliding_window > 0:
        arange_q = torch.arange(q_len, device=attn_weights.device)
        arange_k = torch.arange(kv_seq_len, device=attn_weights.device)
        distance = arange_q.unsqueeze(-1) - arange_k.unsqueeze(0)
        local_mask = (distance >= sliding_window).unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights.masked_fill(local_mask, torch.finfo(attn_weights.dtype).min)

    min_value = torch.finfo(attn_weights.dtype).min
    attn_weights = torch.max(attn_weights, torch.tensor(min_value, device=attn_weights.device, dtype=attn_weights.dtype))

    logit_cap = getattr(attn_module.config, "attn_logit_softcapping", None)
    if logit_cap is not None and logit_cap > 0:
        cap_tensor = torch.tensor(logit_cap, device=attn_weights.device, dtype=attn_weights.dtype)
        attn_weights = torch.tanh(attn_weights / cap_tensor) * cap_tensor

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, attn_module.hidden_size)
    attn_output = attn_module.o_proj(attn_output)

    return attn_output


def test_quant_attention_matches_reference_with_softcapping():
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config.attn_logit_softcapping = 2.0
    config.sliding_window = 2

    reference_attn = modeling_llama.LlamaAttention(config)
    quant_attention = QuantLlamaAttention(reference_attn, config, _make_quant_args())

    hidden_states = torch.randn(1, 4, config.hidden_size)

    reference_output = _reference_attention(reference_attn, hidden_states)
    quant_output, _, _ = quant_attention(hidden_states)

    assert torch.allclose(quant_output, reference_output, atol=1e-6)
