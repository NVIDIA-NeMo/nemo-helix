# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("indexes", [(), ("wheels-v6",), ("wheels-v7",), ("wheels-v6", "wheels-v7")])
def test_rl_cache_cleanup_preserves_referenced_archives(tmp_path: Path, indexes: tuple[str, ...]) -> None:
    script_path = Path(__file__).parents[2] / "rl/scripts/prune-unreferenced-uv-archives.sh"
    script = script_path.read_text()
    environments = []
    for name in ("nemo_rl_venv", "ray_venvs", "gym_venvs"):
        environment = tmp_path / name
        environment.mkdir()
        environments.append(environment)
        script = script.replace(f"/opt/{name}", str(environment))
    cache = tmp_path / "cache"
    archives = {}
    for key in ("obsolete", "referenced", "unrelated"):
        archive = cache / "archive-v0" / key
        distribution = "wandb" if key != "unrelated" else "unrelated"
        (archive / f"{distribution}-1.0.dist-info").mkdir(parents=True)
        archives[key] = archive
    mooncake = cache / "archive-v0/mooncake"
    (mooncake / "mooncake_transfer_engine_cuda13-1.0.dist-info").mkdir(parents=True)
    archives["mooncake"] = mooncake
    (environments[0] / "wandb").symlink_to(archives["referenced"])
    for name in indexes:
        index = cache / name / "pypi"
        index.mkdir(parents=True)
        for key, archive in archives.items():
            (index / key).symlink_to(archive)

    subprocess.run(
        ["bash", "-eu", "-c", script],
        env=dict(os.environ, UV_CACHE_DIR=str(cache)),
        check=True,
        capture_output=True,
        text=True,
    )

    for key, archive in archives.items():
        # Anything no venv symlink points at is removed, including archives the
        # old name-specific loops used to leave in place.
        retained = key == "referenced"
        assert archive.exists() is retained
        for name in indexes:
            link = cache / name / "pypi" / key
            assert link.is_symlink() is retained
            assert link.exists() is retained


def test_prune_drops_files_v0_blobs_with_no_archive_name(tmp_path: Path) -> None:
    script_path = Path(__file__).parents[2] / "rl/scripts/prune-unreferenced-uv-archives.sh"
    script = script_path.read_text()
    venv = tmp_path / "nemo_rl_venv"
    venv.mkdir()
    for name in ("nemo_rl_venv", "ray_venvs", "gym_venvs"):
        script = script.replace(f"/opt/{name}", str(tmp_path / name))
    cache = tmp_path / "cache"
    kept_archive = cache / "archive-v0" / "referenced"
    dropped_archive = cache / "archive-v0" / "obsolete"
    kept_archive.mkdir(parents=True)
    dropped_archive.mkdir(parents=True)
    (venv / "pkg").symlink_to(kept_archive)

    def blob(digest: str) -> Path:
        path = cache / "files-v0" / digest[:2] / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(digest.encode())
        return path

    kept = blob("aa" + "1" * 62)
    dropped = blob("bb" + "2" * 62)
    orphan = blob("cc" + "3" * 62)
    (kept_archive / "kept").hardlink_to(kept)
    (dropped_archive / "dropped").hardlink_to(dropped)

    subprocess.run(
        ["bash", "-eu", "-c", script],
        env=dict(os.environ, UV_CACHE_DIR=str(cache)),
        check=True,
        capture_output=True,
        text=True,
    )

    assert kept.exists()
    assert (kept_archive / "kept").exists()
    assert kept.stat().st_nlink >= 2
    assert not dropped.exists()
    assert not orphan.exists()
    assert not dropped.parent.exists()
    assert (cache / "files-v0").is_dir()


@pytest.mark.parametrize(
    "script_name",
    ["warm-uv-cache.sh", "sync-worker-venvs.sh"],
    ids=["cache-warming", "worker-venvs"],
)
def test_rl_syncs_skip_unused_mooncake_wheels(script_name: str) -> None:
    """Mooncake stays out of every uv sync. The names live in the Dockerfile."""
    dockerfile = (Path(__file__).parents[2] / "rl/Dockerfile.nhx-rl-base").read_text()
    marker = "ARG NO_INSTALL_PACKAGES='"
    start = dockerfile.index(marker) + len(marker)
    packages = json.loads(dockerfile[start : dockerfile.index("'", start)].replace("\\\n", "\n"))
    assert {"mooncake-transfer-engine", "mooncake-transfer-engine-cuda13"} <= set(packages)

    content = (Path(__file__).parents[2] / "rl/scripts" / script_name).read_text()
    syncs = [line for line in content.splitlines() if line.strip().startswith("uv sync ")]
    assert syncs
    assert all('"${skip_args[@]}"' in line for line in syncs)
