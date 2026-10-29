# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nemo_automodel_plugin.jobs.jobs import AutomodelJob
from nhx.customization_common.dry_run import check_local_dataset_rows

_CHAT = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}


def _spec() -> Any:
    return SimpleNamespace(
        dataset=SimpleNamespace(training="ws/data"),
        training=SimpleNamespace(training_type="sft"),
    )


def test_checks_the_training_fileset_with_the_training_validator(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps(_CHAT) + "\n")
    (tmp_path / "validation.jsonl").write_text(json.dumps({"prompt": "q", "completion": "a"}) + "\n")
    [dataset] = AutomodelJob._dry_run_datasets(_spec(), "ws")

    assert dataset.fileset == "ws/data"
    check = check_local_dataset_rows(dataset, tmp_path)
    assert check.passed, check.message
    assert "train.jsonl, validation.jsonl" in check.message


def test_a_row_training_rejects_fails(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps({"instruction": "q", "response": "a"}) + "\n")
    [dataset] = AutomodelJob._dry_run_datasets(_spec(), "ws")

    check = check_local_dataset_rows(dataset, tmp_path)
    assert not check.passed
    assert "train.jsonl line 1 does not conform to the expected schema" in check.message
