import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import load_config_with_rope_fix


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

    assert loaded_config.rope_scaling == rope_scaling


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

    assert loaded_config.rope_scaling["type"] == "yarn"
    for key, value in rope_scaling.items():
        assert loaded_config.rope_scaling[key] == value
