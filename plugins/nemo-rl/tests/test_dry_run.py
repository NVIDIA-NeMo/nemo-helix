# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nemo_rl_plugin.jobs.jobs import RlJob
from nhx.customization_common.dry_run import check_local_dataset_rows


def _spec(training_type: str) -> Any:
    return SimpleNamespace(dataset="ws/data", training=SimpleNamespace(type=training_type))


def test_dpo_rows_pass_and_sft_rows_fail(tmp_path: Path) -> None:
    [dataset] = RlJob._dry_run_datasets(_spec("dpo"), "ws")
    assert dataset.fileset == "ws/data"

    (tmp_path / "training.jsonl").write_text(json.dumps({"prompt": "q", "chosen": "good", "rejected": "bad"}) + "\n")
    assert check_local_dataset_rows(dataset, tmp_path).passed

    (tmp_path / "training.jsonl").write_text(json.dumps({"prompt": "q", "completion": "a"}) + "\n")
    check = check_local_dataset_rows(dataset, tmp_path)
    assert not check.passed
    assert "training.jsonl line 1" in check.message
