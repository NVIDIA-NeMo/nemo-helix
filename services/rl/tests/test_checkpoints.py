# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locating and publishing LoRA adapters and full-weight HF trees."""

import json
from pathlib import Path

import pytest
from nmp.rl.tasks.training.backends.nemo_rl.checkpoints import (
    copy_hf_full_weights,
    copy_lora_adapter,
    find_dcp_weights_root,
    find_hf_full_weight_root,
    find_lora_adapter_root,
    is_peft_publication,
)


def _write_adapter(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (directory / "adapter_model.safetensors").write_text("weights")
    return directory


def _write_tokenizer(checkpoint: Path) -> Path:
    tokenizer = checkpoint / "policy" / "tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "tokenizer_config.json").write_text("{}")
    (tokenizer / "tokenizer.json").write_text("{}")
    return tokenizer


def test_finds_adapter_in_dtensor_v2_layout(tmp_path: Path):
    """Automodel writes model artifacts to <weights_path>/model, so V2 nests one deeper."""
    checkpoint = tmp_path / "step_20"
    adapter = _write_adapter(checkpoint / "policy" / "weights" / "model")

    assert find_lora_adapter_root(checkpoint) == adapter


def test_finds_adapter_in_dtensor_v1_layout(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    adapter = _write_adapter(checkpoint / "policy" / "weights")

    assert find_lora_adapter_root(checkpoint) == adapter


def test_finds_adapter_at_checkpoint_root(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    _write_adapter(checkpoint)

    assert find_lora_adapter_root(checkpoint) == checkpoint


def test_prefers_the_most_specific_layout(tmp_path: Path):
    """A stray root-level config must not shadow the real adapter under policy/weights."""
    checkpoint = tmp_path / "step_20"
    _write_adapter(checkpoint)
    nested = _write_adapter(checkpoint / "policy" / "weights" / "model")

    assert find_lora_adapter_root(checkpoint) == nested


def test_returns_none_for_a_full_weight_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    weights = checkpoint / "policy" / "weights"
    weights.mkdir(parents=True)
    (weights / ".metadata").write_text("dcp")

    assert find_lora_adapter_root(checkpoint) is None


def test_copy_publishes_only_the_adapter_and_adds_the_tokenizer(tmp_path: Path):
    """Optimizer state is a training artifact and must not reach the published model."""
    checkpoint = tmp_path / "step_20"
    adapter = _write_adapter(checkpoint / "policy" / "weights" / "model")
    _write_tokenizer(checkpoint)
    optimizer = checkpoint / "policy" / "optimizer"
    optimizer.mkdir(parents=True)
    (optimizer / "optim.pt").write_text("optimizer state")

    output = tmp_path / "output"
    copy_lora_adapter(checkpoint, adapter, output)

    assert (output / "adapter_config.json").is_file()
    assert (output / "adapter_model.safetensors").is_file()
    assert (output / "tokenizer_config.json").is_file()
    assert not (output / "policy").exists()
    assert not (output / "optim.pt").exists()


def test_copy_keeps_a_tokenizer_already_beside_the_adapter(tmp_path: Path):
    """When Automodel already wrote one, the checkpoint-level copy must not clobber it."""
    checkpoint = tmp_path / "step_20"
    adapter = _write_adapter(checkpoint / "policy" / "weights" / "model")
    (adapter / "tokenizer_config.json").write_text('{"source": "adapter"}')
    tokenizer = _write_tokenizer(checkpoint)
    (tokenizer / "tokenizer_config.json").write_text('{"source": "checkpoint"}')

    output = tmp_path / "output"
    copy_lora_adapter(checkpoint, adapter, output)

    assert '"source": "adapter"' in (output / "tokenizer_config.json").read_text()


def test_copy_without_a_tokenizer_still_publishes_the_adapter(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    adapter = _write_adapter(checkpoint / "policy" / "weights" / "model")

    output = tmp_path / "output"
    copy_lora_adapter(checkpoint, adapter, output)

    assert (output / "adapter_config.json").is_file()
    assert not (output / "tokenizer_config.json").exists()


def _write_hf_shard(directory: Path, name: str = "shard-00001-model-00001-of-00001.safetensors") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("weights")
    return directory


def test_finds_consolidated_full_weight_layout(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    consolidated = checkpoint / "policy" / "weights" / "model" / "consolidated"
    consolidated.mkdir(parents=True)
    (consolidated / "model.safetensors").write_text("consolidated-weights")
    (consolidated / "config.json").write_text('{"model_type": "qwen3"}')
    _write_hf_shard(checkpoint / "policy" / "weights" / "model")

    assert find_hf_full_weight_root(checkpoint) == consolidated


def test_empty_consolidated_dir_falls_back_to_shards(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    (checkpoint / "policy" / "weights" / "model" / "consolidated").mkdir(parents=True)
    model_dir = _write_hf_shard(checkpoint / "policy" / "weights" / "model")

    assert find_hf_full_weight_root(checkpoint) == model_dir


def test_finds_automodel_sharded_full_weight_layout(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    model_dir = _write_hf_shard(checkpoint / "policy" / "weights" / "model")

    assert find_hf_full_weight_root(checkpoint) == model_dir
    assert find_dcp_weights_root(checkpoint) is None


def test_dcp_full_weight_is_not_treated_as_huggingface(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    weights = checkpoint / "policy" / "weights"
    weights.mkdir(parents=True)
    (weights / ".metadata").write_text("dcp")
    (weights / "model").mkdir()
    (weights / "model" / "__0_0.distcp").write_text("shard")

    assert find_hf_full_weight_root(checkpoint) is None
    assert find_dcp_weights_root(checkpoint) == weights


def test_lora_adapter_is_not_a_full_weight_root(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    _write_adapter(checkpoint / "policy" / "weights" / "model")

    assert find_hf_full_weight_root(checkpoint) is None


def test_copy_full_weights_promotes_single_gpu_shard_and_flattens_metadata(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    model_dir = _write_hf_shard(checkpoint / "policy" / "weights" / "model")
    metadata = model_dir / ".hf_metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text(json.dumps({"architectures": ["FSDPQwen3ForCausalLM"]}))
    (metadata / "fqn_to_file_index_mapping.json").write_text("{}")
    _write_tokenizer(checkpoint)
    optimizer = checkpoint / "policy" / "optimizer"
    optimizer.mkdir(parents=True)
    (optimizer / "optim.pt").write_text("optimizer state")

    output = tmp_path / "output"
    copy_hf_full_weights(checkpoint, model_dir, output)

    assert (output / "model.safetensors").is_file()
    assert not (output / "shard-00001-model-00001-of-00001.safetensors").exists()
    assert json.loads((output / "config.json").read_text())["architectures"] == ["Qwen3ForCausalLM"]
    assert (output / "tokenizer_config.json").is_file()
    assert not (output / ".hf_metadata").exists()
    assert not (output / "fqn_to_file_index_mapping.json").exists()
    assert not (output / "optim.pt").exists()


def test_copy_full_weights_prefers_consolidated_export(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    _write_hf_shard(checkpoint / "policy" / "weights" / "model")
    consolidated = checkpoint / "policy" / "weights" / "model" / "consolidated"
    consolidated.mkdir(parents=True)
    (consolidated / "model.safetensors").write_text("consolidated-weights")
    (consolidated / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    _write_tokenizer(checkpoint)

    output = tmp_path / "output"
    copy_hf_full_weights(checkpoint, consolidated, output)

    assert (output / "model.safetensors").read_text() == "consolidated-weights"
    assert (output / "config.json").is_file()
    assert (output / "tokenizer_config.json").is_file()


def test_copy_rejects_multi_rank_shards(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    model_dir = checkpoint / "policy" / "weights" / "model"
    _write_hf_shard(model_dir, "shard-00001-model-00001-of-00001.safetensors")
    _write_hf_shard(model_dir, "shard-00002-model-00001-of-00001.safetensors")

    with pytest.raises(ValueError, match="save_consolidated"):
        copy_hf_full_weights(checkpoint, model_dir, tmp_path / "output")


def test_copy_rejects_multi_file_shards(tmp_path: Path):
    checkpoint = tmp_path / "step_20"
    model_dir = checkpoint / "policy" / "weights" / "model"
    _write_hf_shard(model_dir, "shard-00001-model-00001-of-00002.safetensors")
    _write_hf_shard(model_dir, "shard-00001-model-00002-of-00002.safetensors")

    with pytest.raises(ValueError, match="save_consolidated"):
        copy_hf_full_weights(checkpoint, model_dir, tmp_path / "output")


def test_copy_keeps_shard_names_when_an_index_maps_them(tmp_path: Path):
    """An index resolves tensors to files, so the tree loads and the names must not move."""
    checkpoint = tmp_path / "step_20"
    model_dir = checkpoint / "policy" / "weights" / "model"
    _write_hf_shard(model_dir, "shard-00001-model-00001-of-00002.safetensors")
    _write_hf_shard(model_dir, "shard-00001-model-00002-of-00002.safetensors")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 14},
                "weight_map": {
                    "a.weight": "shard-00001-model-00001-of-00002.safetensors",
                    "b.weight": "shard-00001-model-00002-of-00002.safetensors",
                },
            }
        )
    )

    output = tmp_path / "output"
    copy_hf_full_weights(checkpoint, model_dir, output)

    assert (output / "shard-00001-model-00001-of-00002.safetensors").is_file()
    assert (output / "shard-00001-model-00002-of-00002.safetensors").is_file()
    assert (output / "model.safetensors.index.json").is_file()


def test_copy_keeps_a_single_shard_named_by_an_index(tmp_path: Path):
    """Even the promotable layout must stay put when an index already names the file."""
    checkpoint = tmp_path / "step_20"
    model_dir = checkpoint / "policy" / "weights" / "model"
    _write_hf_shard(model_dir, "shard-00001-model-00001-of-00001.safetensors")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {"a.weight": "shard-00001-model-00001-of-00001.safetensors"},
            }
        )
    )

    output = tmp_path / "output"
    copy_hf_full_weights(checkpoint, model_dir, output)

    assert (output / "shard-00001-model-00001-of-00001.safetensors").is_file()
    assert not (output / "model.safetensors").exists()


def test_peft_publication_requires_an_adapter_on_disk():
    adapter = Path("/tmp/adapter")
    assert is_peft_publication(requested_lora=True, adapter_root=adapter) is True
    assert is_peft_publication(requested_lora=True, adapter_root=None) is False
    assert is_peft_publication(requested_lora=False, adapter_root=adapter) is False
    assert is_peft_publication(requested_lora=False, adapter_root=None) is False
