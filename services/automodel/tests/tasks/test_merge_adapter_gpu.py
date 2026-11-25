# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real LoRA merge on a GPU, with a real base checkpoint and adapter. Never runs in CI: it is marked
``skip_in_ci`` and needs a GPU plus a checkpoint, so it skips itself unless both env vars are set::

    NHX_MERGE_TEST_BASE_MODEL=/path/to/base NHX_MERGE_TEST_ADAPTER=/path/to/adapter \\
        pytest services/automodel/tests/tasks/test_merge_adapter_gpu.py -m skip_in_ci

``NHX_MERGE_TEST_TRUST_REMOTE_CODE=1`` loads the base with its remote code."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from nhx.automodel.tasks.merge_adapter import MergeAdapterStepConfig, run_merge  # noqa: E402
from safetensors import safe_open  # noqa: E402

BASE = os.environ.get("NHX_MERGE_TEST_BASE_MODEL")
ADAPTER = os.environ.get("NHX_MERGE_TEST_ADAPTER")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skip_in_ci,
    pytest.mark.skipif(not (BASE and ADAPTER), reason="set NHX_MERGE_TEST_BASE_MODEL and NHX_MERGE_TEST_ADAPTER"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


def _tensor_names(checkpoint: Path) -> dict[str, Path]:
    names = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            names.update({name: shard for name in handle.keys()})  # noqa: SIM118
    return names


def _load(names: dict[str, Path], name: str):
    with safe_open(names[name], framework="pt") as handle:
        return handle.get_tensor(name)


def test_merging_a_real_adapter_changes_the_base_weights_and_leaves_a_loadable_checkpoint(tmp_path: Path) -> None:
    base, adapter, output = Path(str(BASE)), Path(str(ADAPTER)), tmp_path / "merged"
    base_config = json.loads((base / "config.json").read_text())

    run_merge(
        MergeAdapterStepConfig(
            has_vision_tower="vision_config" in base_config,
            trust_remote_code=os.environ.get("NHX_MERGE_TEST_TRUST_REMOTE_CODE") == "1",
        ),
        base=base, adapter=adapter, output=output,
    )  # fmt: skip

    merged_config = json.loads((output / "config.json").read_text())
    assert merged_config["architectures"] == base_config["architectures"]
    assert (output / "tokenizer_config.json").exists()
    merged, original = _tensor_names(output), _tensor_names(base)
    assert not [name for name in merged if "lora_" in name], "adapter tensors were left in the merged checkpoint"

    with safe_open(next(adapter.glob("*.safetensors")), framework="pt") as handle:
        modules = sorted({key.removeprefix("base_model.model.").split(".lora_")[0] for key in handle.keys()})  # noqa: SIM118
    changed = []
    for module in modules:
        for suffix in (".weight", ".linear.weight"):
            if module + suffix in merged and module + suffix in original:
                delta = _load(merged, module + suffix).float() - _load(original, module + suffix).float()
                changed.append(delta.abs().max().item() > 0)
    assert changed, f"none of the adapter's modules ({modules[:3]}...) are in both checkpoints"
    assert any(changed), (
        f"none of the {len(changed)} adapter modules found in both checkpoints changed the base weights"
    )
