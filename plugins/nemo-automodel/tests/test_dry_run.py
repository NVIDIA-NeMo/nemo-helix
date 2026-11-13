# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_automodel_plugin.jobs import jobs as jobs_mod
from nemo_automodel_plugin.jobs.jobs import AutomodelJob
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.models.types import ModelSpec
from nhx.customization_common.dry_run import check_local_dataset_rows
from nhx.customization_common.schemas.checkpoint import CheckpointTraits

_CHAT = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
_PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZ1kAAAAASUVORK5CYII="
)


def _image_chat(url: str) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "q"}],
            },
            {"role": "assistant", "content": "a"},
        ]
    }


def _spec() -> Any:
    return SimpleNamespace(
        dataset=SimpleNamespace(training="ws/data"),
        training=SimpleNamespace(training_type="sft"),
        model="ws/vlm",
    )


def test_checks_the_training_fileset_with_the_training_validator(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps(_CHAT) + "\n")
    (tmp_path / "validation.jsonl").write_text(json.dumps({"prompt": "q", "completion": "a"}) + "\n")
    [dataset] = AutomodelJob._dry_run_datasets(_spec(), "ws", CheckpointTraits())

    assert dataset.fileset == "ws/data"
    check = check_local_dataset_rows(dataset, tmp_path)
    assert check.passed, check.message
    assert "train.jsonl, validation.jsonl" in check.message


def test_a_row_training_rejects_fails(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps({"instruction": "q", "response": "a"}) + "\n")
    [dataset] = AutomodelJob._dry_run_datasets(_spec(), "ws", CheckpointTraits())

    check = check_local_dataset_rows(dataset, tmp_path)
    assert not check.passed
    assert "train.jsonl line 1 does not conform to the expected schema" in check.message


@pytest.mark.parametrize("url", ["a.png", _PNG_DATA_URL], ids=["dataset-path", "data-url"])
@pytest.mark.parametrize("has_vision_tower", [True, False])
def test_an_image_row_passes_only_for_a_vision_model(tmp_path: Path, has_vision_tower: bool, url: str) -> None:
    """Dry-run accepts an image row, by path or data URL, only when the model has a vision tower."""
    (tmp_path / "train.jsonl").write_text(json.dumps(_image_chat(url)) + "\n")
    traits = CheckpointTraits(has_vision_tower=has_vision_tower)
    [dataset] = AutomodelJob._dry_run_datasets(_spec(), "ws", traits)

    check = check_local_dataset_rows(dataset, tmp_path)
    assert check.passed is has_vision_tower, check.message
    if not has_vision_tower:
        assert "train.jsonl line 1 has an image row, which requires a model with a vision tower" in check.message


@pytest.mark.parametrize("has_vision_tower", [True, False])
async def test_dry_run_reads_the_traits_of_the_model_entity(
    monkeypatch: pytest.MonkeyPatch, has_vision_tower: bool
) -> None:
    async def fetch_model_entity(model_ref: str, workspace: str, platform: object) -> SimpleNamespace:
        assert (model_ref, workspace) == ("ws/vlm", "ws")
        return SimpleNamespace(spec=ModelSpec.model_construct(has_vision_tower=has_vision_tower))

    monkeypatch.setattr(jobs_mod, "fetch_model_entity", fetch_model_entity)
    traits = await AutomodelJob._dry_run_traits(_spec(), "ws", AsyncNemoClient(base_url="http://platform.test"))

    assert traits.has_vision_tower is has_vision_tower
