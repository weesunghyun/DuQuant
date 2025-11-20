import json
from pathlib import Path

from utils import load_config_with_rope_fix


def test_load_config_with_rope_type_preserved(tmp_path: Path):
    # Simulate a Llama-3 style config that includes a rope_type value rejected
    # by older ``transformers`` releases.
    config_dir = tmp_path / "llama3-config"
    config_dir.mkdir()

    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
        },
    }

    (config_dir / "config.json").write_text(json.dumps(config))

    loaded_config = load_config_with_rope_fix(str(config_dir))

    assert loaded_config.rope_scaling["rope_type"] == "llama3"
    assert loaded_config.rope_scaling["type"] == "dynamic"
    assert loaded_config.rope_scaling["factor"] == 8.0
    assert (
        loaded_config.rope_scaling["original_max_position_embeddings"]
        == 8192
    )
