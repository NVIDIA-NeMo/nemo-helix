# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nmp.customization_common.tasks.file_io_utils import list_local_files, validate_safe_path


def test_list_local_files_skips_unreadable_entries(tmp_path: Path) -> None:
    root = tmp_path / "output_model"
    root.mkdir()
    (root / "adapter_model.safetensors").write_bytes(b"x" * 8)
    wandb_dir = root / "wandb" / "run-1" / "logs"
    wandb_dir.mkdir(parents=True)
    (wandb_dir / "debug.log").write_text("ok")
    broken = wandb_dir / "debug-core.log"
    broken.symlink_to(root / "missing-target")

    files = list_local_files(root)

    paths = {f.path for f in files}
    assert "adapter_model.safetensors" in paths
    assert "wandb/run-1/logs/debug.log" in paths
    assert "wandb/run-1/logs/debug-core.log" not in paths
    assert len(files) == 2


def test_list_local_files_single_file(tmp_path: Path) -> None:
    file_path = tmp_path / "weights.bin"
    file_path.write_bytes(b"abc")

    files = list_local_files(file_path)

    assert len(files) == 1
    assert files[0].path == "weights.bin"
    assert files[0].size == 3


def test_validate_safe_path_remaps_legacy_job_storage_root(tmp_path: Path) -> None:
    storage = tmp_path / "job"
    target = storage / "output_model"
    target.mkdir(parents=True)

    resolved = validate_safe_path(storage, "/run/scratch/job/output_model")

    assert resolved == target.resolve()


def test_validate_safe_path_rejects_unknown_absolute_path(tmp_path: Path) -> None:
    from nmp.customization_common.schemas.file_io import PathTraversalError

    storage = tmp_path / "job"
    storage.mkdir()

    with pytest.raises(PathTraversalError):
        validate_safe_path(storage, "/etc/passwd")
