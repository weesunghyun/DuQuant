from types import SimpleNamespace

import pytest
import torch

from models.LMClass import LMClass
from models.models_utils import get_rolling_token_windows, make_disjoint_window


class DummyConfig:
    def __init__(self, max_position_embeddings: int = 32):
        self.max_position_embeddings = max_position_embeddings


class DummyTokenizer:
    vocab_size = 256

    def __init__(self):
        self.bos_token_id = 11
        self.eos_token_id = 12
        self.default_system_prompt = "Be helpful"
        self.chat_template = "llama-3"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 50 + 20 for ch in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        tokens = [self.bos_token_id]
        for message in messages:
            tokens.append(99)
            tokens.extend(self.encode(message["content"], add_special_tokens=False))
        if add_generation_prompt:
            tokens.append(98)
        return tokens if tokenize else ""

    def batch_decode(self, tokens, skip_special_tokens=True):
        return ["" for _ in tokens]

    def __call__(self, strings, padding=True, add_special_tokens=False, return_tensors=None):
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in strings]
        max_len = max(len(ids) for ids in encoded)
        padded = [ids + [0] * (max_len - len(ids)) for ids in encoded]
        return {"input_ids": torch.tensor(padded)}


class DummyModel:
    def __init__(self):
        self.config = DummyConfig()

    def eval(self):
        return self

    def __call__(self, input_ids):
        vocab = 256
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, vocab)
        for batch_id in range(batch):
            for token_idx in range(seq):
                token_id = input_ids[batch_id, token_idx]
                logits[batch_id, token_idx, token_id] = 5.0
        return {"logits": logits}


def test_llama3_perplexity_uses_chat_template(monkeypatch):
    dummy_tokenizer = DummyTokenizer()
    dummy_model = DummyModel()

    monkeypatch.setattr(
        "models.LMClass.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: dummy_tokenizer,
    )
    monkeypatch.setattr(
        "models.LMClass.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: dummy_model,
    )
    monkeypatch.setattr(
        "models.LMClass.load_config_with_rope_fix",
        lambda *args, **kwargs: DummyConfig(),
    )

    args = SimpleNamespace(model="Meta-Llama-3-Test", batch_size=1, attn_implementation=None)
    lm = LMClass(args)
    snippet = "chat template check"

    ppl_with_template = lm.loglikelihood_rolling([(snippet,)])[0]

    messages = [
        {"role": "system", "content": dummy_tokenizer.default_system_prompt},
        {"role": "user", "content": snippet},
    ]
    template_tokens = dummy_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    rolling_windows = list(
        map(
            make_disjoint_window,
            get_rolling_token_windows(
                token_list=template_tokens,
                prefix_token=dummy_tokenizer.bos_token_id,
                max_seq_len=lm.max_length,
                context_len=1,
            ),
        )
    )
    rolling_windows = [(None,) + window for window in rolling_windows]
    reference_total = sum(
        value[0] for value in lm._loglikelihood_tokens(rolling_windows, disable_tqdm=True)
    )

    raw_tokens = dummy_tokenizer.encode(snippet, add_special_tokens=False)
    raw_windows = list(
        map(
            make_disjoint_window,
            get_rolling_token_windows(
                token_list=raw_tokens,
                prefix_token=dummy_tokenizer.eos_token_id,
                max_seq_len=lm.max_length,
                context_len=1,
            ),
        )
    )
    raw_windows = [(None,) + window for window in raw_windows]
    raw_total = sum(
        value[0] for value in lm._loglikelihood_tokens(raw_windows, disable_tqdm=True)
    )

    assert ppl_with_template == pytest.approx(reference_total)
    assert ppl_with_template != pytest.approx(raw_total)
