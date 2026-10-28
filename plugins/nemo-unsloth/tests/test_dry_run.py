# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nemo_unsloth_plugin.jobs.jobs import UnslothJob
from nhx.customization_common.dry_run import check_local_dataset_rows


def _spec(*, apply_chat_template: bool, validation_path: str | None = None) -> Any:
    return SimpleNamespace(
        dataset=SimpleNamespace(
            path="ws/train",
            validation_path=validation_path,
            apply_chat_template=apply_chat_template,
            text_field="text",
        )
    )


def test_checks_training_and_validation_filesets(tmp_path: Path) -> None:
    datasets = UnslothJob._dry_run_datasets(_spec(apply_chat_template=False, validation_path="ws/val"), "ws")
    assert [(dataset.name, dataset.fileset) for dataset in datasets] == [
        ("training dataset", "ws/train"),
        ("validation dataset", "ws/val"),
    ]


def test_chat_template_needs_messages_rows(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps({"text": "rendered"}) + "\n")
    [dataset] = UnslothJob._dry_run_datasets(_spec(apply_chat_template=True), "ws")

    check = check_local_dataset_rows(dataset, tmp_path)
    assert not check.passed
    assert "line 1 needs a 'messages' list" in check.message
