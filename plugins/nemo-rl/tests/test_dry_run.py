# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.models.types import ModelSpec
from nemo_rl_plugin.jobs import jobs as jobs_mod
from nemo_rl_plugin.jobs.jobs import RlJob
from nhx.customization_common.dry_run import check_local_dataset_rows
from nhx.customization_common.schemas.checkpoint import CheckpointTraits

_IMAGE_PROMPT = [
    {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "a.png"}}, {"type": "text", "text": "q"}],
    }
]
_IMAGE_DPO_ROW = {"prompt": _IMAGE_PROMPT, "chosen": "good", "rejected": "bad"}
_IMAGE_GRPO_ROW = {
    "responses_create_params": {"input": _IMAGE_PROMPT},
    "agent_ref": {"name": "verifiers_agent"},
}


def _spec(training_type: str) -> Any:
    return SimpleNamespace(model="ws/vlm", dataset="ws/data", training=SimpleNamespace(type=training_type))


def test_dpo_rows_pass_and_sft_rows_fail(tmp_path: Path) -> None:
    [dataset] = RlJob._dry_run_datasets(_spec("dpo"), "ws", CheckpointTraits())
    assert dataset.fileset == "ws/data"

    (tmp_path / "training.jsonl").write_text(json.dumps({"prompt": "q", "chosen": "good", "rejected": "bad"}) + "\n")
    assert check_local_dataset_rows(dataset, tmp_path).passed

    (tmp_path / "training.jsonl").write_text(json.dumps({"prompt": "q", "completion": "a"}) + "\n")
    check = check_local_dataset_rows(dataset, tmp_path)
    assert not check.passed
    assert "training.jsonl line 1" in check.message


@pytest.mark.parametrize(
    ("training_type", "row", "error"),
    [
        ("grpo", _IMAGE_GRPO_ROW, None),
        ("dpo", _IMAGE_DPO_ROW, "training.jsonl line 1 contains an image. DPO on a vision checkpoint"),
    ],
)
def test_image_rows_on_a_vision_model_are_checked_as_training_checks_them(
    tmp_path: Path, training_type: str, row: dict, error: str | None
) -> None:
    (tmp_path / "training.jsonl").write_text(json.dumps(row) + "\n")
    [dataset] = RlJob._dry_run_datasets(_spec(training_type), "ws", CheckpointTraits(has_vision_tower=True))

    check = check_local_dataset_rows(dataset, tmp_path)
    assert check.passed is (error is None), check.message
    if error is not None:
        assert error in check.message


@pytest.mark.parametrize("has_vision_tower", [True, False])
async def test_dry_run_reads_the_traits_of_the_model_entity(
    monkeypatch: pytest.MonkeyPatch, has_vision_tower: bool
) -> None:
    async def fetch_model_entity(model_ref: str, workspace: str, platform: object) -> SimpleNamespace:
        assert (model_ref, workspace) == ("ws/vlm", "ws")
        return SimpleNamespace(spec=ModelSpec.model_construct(has_vision_tower=has_vision_tower))

    monkeypatch.setattr(jobs_mod, "fetch_model_entity", fetch_model_entity)
    traits = await RlJob._dry_run_traits(_spec("dpo"), "ws", AsyncNemoClient(base_url="http://platform.test"))

    assert traits.has_vision_tower is has_vision_tower
