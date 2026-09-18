# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.taskset_sources import (
    TasksetSourceMaterialization,
)
from pydantic import ValidationError

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def _materialization_payload(root: Path) -> dict[str, object]:
    return {
        "source_uri": "harbor://tasksets/example@latest",
        "materialized_root": root,
        "revision_digest": _DIGEST_A,
    }


def test_materialization_validates_and_round_trips_all_fields(tmp_path: Path) -> None:
    receipt = TasksetSourceMaterialization.model_validate(_materialization_payload(tmp_path))

    assert TasksetSourceMaterialization.model_validate_json(receipt.model_dump_json()) == receipt
    assert receipt.materialized_root == tmp_path


@pytest.mark.parametrize(
    "field",
    ["source_uri", "materialized_root", "revision_digest"],
)
def test_materialization_requires_every_field(tmp_path: Path, field: str) -> None:
    payload = _materialization_payload(tmp_path)
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        TasksetSourceMaterialization.model_validate(payload)


@pytest.mark.parametrize(
    "source_uri",
    ["", "tasksets/example", "/tasksets/example", "harbor://tasksets/example with space"],
)
def test_materialization_rejects_non_absolute_source_uri(tmp_path: Path, source_uri: str) -> None:
    payload = _materialization_payload(tmp_path)
    payload["source_uri"] = source_uri

    with pytest.raises(ValidationError, match="absolute URI"):
        TasksetSourceMaterialization.model_validate(payload)


def test_materialization_accepts_none_revision_and_empty_members(tmp_path: Path) -> None:
    payload = _materialization_payload(tmp_path)
    payload["revision_digest"] = None

    receipt = TasksetSourceMaterialization.model_validate(payload)

    assert receipt.revision_digest is None


def test_materialization_rejects_relative_materialized_root() -> None:
    payload = _materialization_payload(Path("relative/root"))

    with pytest.raises(ValidationError, match="absolute path"):
        TasksetSourceMaterialization.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision_digest", "A" * 64),
        ("revision_digest", "a" * 63),
    ],
)
def test_materialization_rejects_non_sha256_digest_shapes(tmp_path: Path, field: str, value: object) -> None:
    payload = _materialization_payload(tmp_path)
    payload[field] = value

    with pytest.raises(ValidationError):
        TasksetSourceMaterialization.model_validate(payload)


def test_materialization_forbids_extra_fields_and_field_reassignment(tmp_path: Path) -> None:
    payload = _materialization_payload(tmp_path)
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        TasksetSourceMaterialization.model_validate(payload)

    receipt = TasksetSourceMaterialization.model_validate(_materialization_payload(tmp_path))
    with pytest.raises(ValidationError, match="frozen"):
        receipt.source_uri = "harbor://tasksets/other"  # type: ignore[misc]
