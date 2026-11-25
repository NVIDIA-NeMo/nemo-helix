# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge task. ``merge_lora_adapter`` needs torch and real weights, so ``run_merge`` gets a stand-in that
writes the checkpoint a merge would; the tests check what the step does around it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nhx.automodel.tasks.merge_adapter import MergeAdapterStepConfig, run_merge, uses_builtin_model_class
from nhx.automodel.tasks.training.backends.registry import MODEL_ARCH_MAPPING

REGISTERED = next(iter(MODEL_ARCH_MAPPING))


def _base(path: Path, architecture: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    return path


@pytest.mark.parametrize(
    ("architecture", "override", "expected"),
    [(REGISTERED, False, True), (REGISTERED, True, False), ("NotRegistered", False, False)],
)
def test_builtin_class_is_used_for_architectures_automodel_implements(
    tmp_path: Path, architecture: str, override: bool, expected: bool
) -> None:
    assert uses_builtin_model_class(_base(tmp_path, architecture), override_custom_impl=override) is expected


@pytest.mark.parametrize("template", ["{{ m }}", None])
def test_run_merge_gives_the_merge_its_flags_and_applies_the_chat_template(
    tmp_path: Path, template: str | None
) -> None:
    base, adapter, output = _base(tmp_path / "base", REGISTERED), tmp_path / "adapter", tmp_path / "out"
    calls: list[dict] = []

    def merge(**kwargs) -> None:
        calls.append(kwargs)
        kwargs["output_path"].mkdir()
        (kwargs["output_path"] / "tokenizer_config.json").write_text('{"eos_token": "</s>"}')

    run_merge(
        MergeAdapterStepConfig(has_vision_tower=True, trust_remote_code=True, chat_template=template),
        base=base, adapter=adapter, output=output, merge=merge,
    )  # fmt: skip

    assert calls == [
        dict(adapter_path=adapter, base_model_path=str(base), output_path=output, trust_remote_code=True,
             has_vision_tower=True, builtin_model_class=True)
    ]  # fmt: skip
    tokenizer = json.loads((output / "tokenizer_config.json").read_text())
    assert tokenizer.get("chat_template") == template and tokenizer["eos_token"] == "</s>"
