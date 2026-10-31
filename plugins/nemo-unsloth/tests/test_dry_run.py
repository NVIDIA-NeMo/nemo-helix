# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nemo_unsloth_plugin.jobs.jobs import UnslothJob
from nhx.customization_common.dry_run import check_local_dataset_rows


def _spec(*, apply_chat_template: bool) -> Any:
    return SimpleNamespace(
        dataset=SimpleNamespace(path="ws/data", apply_chat_template=apply_chat_template, text_field="text")
    )


def test_checks_the_training_and_validation_files_of_the_one_fileset(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps({"text": "t"}) + "\n")
    (tmp_path / "validation.jsonl").write_text(json.dumps({"bad": "row"}) + "\n")
    [dataset] = UnslothJob._dry_run_datasets(_spec(apply_chat_template=False), "ws")

    assert dataset.fileset == "ws/data"
    check = check_local_dataset_rows(dataset, tmp_path)
    assert not check.passed
    assert "validation.jsonl line 1" in check.message


def test_chat_template_needs_messages_rows(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps({"text": "rendered"}) + "\n")
    [dataset] = UnslothJob._dry_run_datasets(_spec(apply_chat_template=True), "ws")

    check = check_local_dataset_rows(dataset, tmp_path)
    assert not check.passed
    assert "line 1 needs a 'messages' list" in check.message
