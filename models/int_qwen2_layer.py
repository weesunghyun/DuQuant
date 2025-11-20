import torch
from torch import nn
from typing import Optional, Tuple
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.du_norm import DuQwen2RMSNorm
import math
import copy
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.activations import ACT2FN
from models.transformation import *


class QuantQwen2MLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()

        self.gate_proj = QuantLinear(
            org_module.gate_proj,
            args.gate_weight_quant_params,
            args.gate_act_quant_params,
        )
        self.down_proj = QuantLinear(
            org_module.down_proj,
            args.down_weight_quant_params,
            args.down_act_quant_params,
        )
        self.up_proj = QuantLinear(
            org_module.up_proj,
            args.up_weight_quant_params,
            args.up_act_quant_params,
        )
        self.act_fn = ACT2FN[hidden_act]
        self.init_duquant_params = (
            torch.tensor(0)
            if args.gate_weight_quant_params["quant_method"] == "duquant"
            else torch.tensor(1)
        )

    def forward(self, x):
        if not self.init_duquant_params:
            self.init_duquant_params = torch.tensor(1)
            act = self.act_fn(self.gate_proj(x))
            self.up_proj.copy_quantizers_duquant_params(self.gate_proj)
            mul = act * self.up_proj(x)
            return self.down_proj(mul)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantQwen2Attention(nn.Module):
    """Multi-headed attention adapted for Qwen2"""

    def __init__(self, org_module: nn.Module, config: Qwen2Config, args=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = copy.deepcopy(org_module.rotary_emb)

        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.k_weight_quant_params,
            args.k_act_quant_params,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.v_weight_quant_params,
            args.v_act_quant_params,
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.q_weight_quant_params,
            args.q_act_quant_params,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, args.o_weight_quant_params, args.o_act_quant_params
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul, rotate=None
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul, rotate=None
        )

        self.use_weight_quant = False
        self.use_act_quant = False
        self.init_duquant_params = (
            torch.tensor(0)
            if args.gate_weight_quant_params["quant_method"] == "duquant"
            else torch.tensor(1)
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if not self.init_duquant_params:
            self.k_proj.copy_quantizers_duquant_params(self.q_proj)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if not self.init_duquant_params:
            self.v_proj.copy_quantizers_duquant_params(self.q_proj)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        self.init_duquant_params = torch.tensor(1)

        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


class QuantQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, ori_layer: nn.Module, args=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = QuantQwen2Attention(ori_layer.self_attn, config, args)

        self.mlp = QuantQwen2MLP(
            ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
        )

        self.input_layernorm = DuQwen2RMSNorm(
            ori_layer.input_layernorm, eps=ori_layer.input_layernorm.variance_epsilon
        )
        self.post_attention_layernorm = DuQwen2RMSNorm(
            ori_layer.post_attention_layernorm,
            eps=ori_layer.post_attention_layernorm.variance_epsilon,
        )
        self.use_weight_quant = False
        self.use_act_quant = False
        self.let = 0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantLinear):
                m.set_quant_state(weight_quant, act_quant)

    def load_smooth_params(self, duquant_parameters, dev):
        layer_param = duquant_parameters
        self.qkt_smooth_scale = nn.Parameter(layer_param["qkt_smooth_scale"].to(dev))
        for name, module in self.named_modules():
            if "attention" in name:
                if isinstance(module, QuantMatMul):
                    module.register_buffer("qkt_smooth_scale", self.qkt_smooth_scale)
                if isinstance(module, QuantLinear):
                    for key in ["q", "k", "v", "o"]:
                        if key in name:
                            module.register_parameter(f"{key}kv_smooth_shift", nn.Parameter(layer_param[f"{key}kv_smooth_shift"]))
                            module.register_parameter(f"{key}kv_smooth_scale", nn.Parameter(layer_param[f"{key}kv_smooth_scale"]))
            if isinstance(module, QuantLinear):
                for key in ["up", "gate", "down"]:
                    if key in name:
                        module.register_parameter(f"{key}kv_smooth_shift", nn.Parameter(layer_param[f"{key}kv_smooth_shift"]))
                        module.register_parameter(f"{key}kv_smooth_scale", nn.Parameter(layer_param[f"{key}kv_smooth_scale"]))

