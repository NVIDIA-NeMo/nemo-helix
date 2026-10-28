# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("stage", [0, 1], ids=["runtime-packages", "setuptools"])
@pytest.mark.parametrize("indexes", [(), ("wheels-v6",), ("wheels-v7",), ("wheels-v6", "wheels-v7")])
def test_rl_cache_cleanup_preserves_referenced_archives(tmp_path: Path, stage: int, indexes: tuple[str, ...]) -> None:
    dockerfile = Path(__file__).parents[2] / "rl/Dockerfile.nhx-rl-base"
    loop_start = 'for d in "${UV_CACHE_DIR}"/archive-v0/*/; do\n'
    body = dockerfile.read_text().split(loop_start)[stage + 1].split("\ndone\n", 1)[0]
    script = loop_start + body + "\ndone\n"
    environments = []
    for name in ("nemo_rl_venv", "ray_venvs", "gym_venvs"):
        environment = tmp_path / name
        environment.mkdir()
        environments.append(environment)
        script = script.replace(f"/opt/{name}", str(environment))
    cache = tmp_path / "cache"
    package = "wandb" if stage == 0 else "setuptools"
    archives = {}
    for key in ("obsolete", "referenced", "unrelated"):
        archive = cache / "archive-v0" / key
        distribution = package if key != "unrelated" else "unrelated"
        (archive / f"{distribution}-1.0.dist-info").mkdir(parents=True)
        archives[key] = archive
    if stage == 0:
        archive = cache / "archive-v0/mooncake"
        (archive / "mooncake_transfer_engine_cuda13-1.0.dist-info").mkdir(parents=True)
        archives["mooncake"] = archive
    (environments[0] / package).symlink_to(archives["referenced"])
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
        retained = key in {"referenced", "unrelated"}
        assert archive.exists() is retained
        for name in indexes:
            link = cache / name / "pypi" / key
            assert link.is_symlink() is retained
            assert link.exists() is retained


@pytest.mark.parametrize("stage", [0, 1], ids=["cache-warming", "worker-venvs"])
def test_rl_syncs_skip_unused_mooncake_wheels(stage: int) -> None:
    dockerfile = Path(__file__).parents[2] / "rl/Dockerfile.nhx-rl-base"
    content = dockerfile.read_text()
    blocks = content.split("SKIP=(\n")[1:]
    assert len(blocks) == 2, "Keep coverage aligned with both RL sync stages"
    array = "SKIP=(\n" + blocks[stage].split("\n)", 1)[0] + "\n)\n"
    result = subprocess.run(
        ["bash", "-eu", "-c", array + 'printf "%s\\n" "${SKIP[@]}"'],
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = result.stdout.splitlines()
    skipped = {arguments[index + 1] for index, arg in enumerate(arguments) if arg == "--no-install-package"}
    assert {"mooncake-transfer-engine", "mooncake-transfer-engine-cuda13"} <= skipped
    # An array that is never passed to uv would not prevent wheel installation.
    syncs = [line for line in content.splitlines() if line.strip().startswith("uv sync ")]
    assert syncs
    assert all('"${SKIP[@]}"' in line for line in syncs)
