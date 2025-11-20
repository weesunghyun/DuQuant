import inspect
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import load_config_with_rope_fix


def _supported_rope_scaling_types_from_attention():
    supported_types = set()
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention

        for attr in ("_ROPE_TYPES", "ROPE_TYPES"):
            rope_types = getattr(LlamaAttention, attr, None)
            if isinstance(rope_types, (list, tuple, set)):
                supported_types.update(rope_types)

        init_rope = getattr(LlamaAttention, "_init_rope", None)
        if callable(init_rope):
            try:
                source = inspect.getsource(init_rope)
            except (OSError, TypeError):
                source = ""

            for candidate in ("yarn", "linear", "dynamic", "longrope"):
                if f"'{candidate}'" in source or f'"{candidate}"' in source:
                    supported_types.add(candidate)
    except Exception:
        pass

    return supported_types


def test_load_config_with_rope_type_preserved(tmp_path: Path):
    # Simulate a Llama-3 style config that includes rope scaling values rejected
    # by older ``transformers`` releases.
    config_dir = tmp_path / "llama3-config"
    config_dir.mkdir()

    rope_scaling = {
        "type": "yarn",
        "rope_type": "llama3",
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "low_freq_factor": 0.5,
        "high_freq_factor": 1.5,
    }

    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "rope_scaling": rope_scaling,
    }

    (config_dir / "config.json").write_text(json.dumps(config))

    loaded_config = load_config_with_rope_fix(str(config_dir))

    supported_types = _supported_rope_scaling_types_from_attention()
    fallback_order = ["linear", "dynamic", "yarn"]
    expected_type = rope_scaling["type"]
    if supported_types and expected_type not in supported_types:
        expected_type = next(
            (rope_type for rope_type in fallback_order if rope_type in supported_types),
            next(iter(supported_types)),
        )

    assert loaded_config.rope_scaling["type"] == expected_type
    for key, value in rope_scaling.items():
        if key != "type":
            assert loaded_config.rope_scaling[key] == value


def test_load_config_adds_missing_rope_type_field(tmp_path: Path):
    config_dir = tmp_path / "llama3-config-missing-type"
    config_dir.mkdir()

    rope_scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "low_freq_factor": 0.5,
        "high_freq_factor": 1.5,
    }

    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "rope_scaling": rope_scaling,
    }

    (config_dir / "config.json").write_text(json.dumps(config))

    loaded_config = load_config_with_rope_fix(str(config_dir))

    supported_types = _supported_rope_scaling_types_from_attention()
    fallback_order = ["linear", "dynamic", "yarn"]
    expected_type = "yarn"
    if supported_types and expected_type not in supported_types:
        expected_type = next(
            (rope_type for rope_type in fallback_order if rope_type in supported_types),
            next(iter(supported_types)),
        )

    assert loaded_config.rope_scaling["type"] == expected_type
    for key, value in rope_scaling.items():
        assert loaded_config.rope_scaling[key] == value


def test_load_config_sets_supported_rope_type(tmp_path: Path):
    config_dir = tmp_path / "llama3-config-missing-type"
    config_dir.mkdir()

    rope_scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "low_freq_factor": 0.5,
        "high_freq_factor": 1.5,
    }

    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "rope_scaling": rope_scaling,
    }

    (config_dir / "config.json").write_text(json.dumps(config))

    loaded_config = load_config_with_rope_fix(str(config_dir))

    supported_types = _supported_rope_scaling_types_from_attention()
    if supported_types and "yarn" not in supported_types:
        assert loaded_config.rope_scaling["type"] in supported_types

    from transformers.models.llama.modeling_llama import LlamaAttention

    attention = LlamaAttention(loaded_config)

    assert attention.rotary_emb is not None
