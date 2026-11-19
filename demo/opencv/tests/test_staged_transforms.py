# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the montage tasks with staged Filesets and no live services."""

import json
import shutil
import struct
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(find_spec("cv2") is None, reason="Run with uv --with opencv-python-headless==5.1.0.95")


def run_transform(transform: str, directories: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-u", str(DEMO / "demo.py"), transform]
    for role, directory in directories.items():
        command.extend([f"--{role}", str(directory)])
    return subprocess.run(command, text=True, capture_output=True, check=False)


@pytest.mark.parametrize("image_count", [1, 2])
def test_staged_photos_produce_edges_montage_and_captions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image_count: int
) -> None:
    from demo.opencv import captions

    images = tmp_path / "images"
    images.mkdir()
    fixtures = ["sutro-tower.jpg", "birthday-cake.jpg"][:image_count]
    for name in fixtures:
        shutil.copyfile(DEMO / "fixtures" / name, images / name)
    edges, montage, output = (tmp_path / name for name in ("edges", "montage", "captions"))
    detected = run_transform("detect-edges", {"images": images, "edges": edges})
    assert detected.returncode == 0, detected.stderr
    assert "controller will upload and commit" in detected.stdout
    assert {path.name for path in edges.iterdir()} == {Path(name).stem + ".png" for name in fixtures}
    assembled = run_transform("build-montage", {"images": images, "edges": edges, "montage": montage})
    assert assembled.returncode == 0, assembled.stderr
    sheet = (montage / "contact-sheet.png").read_bytes()
    assert sheet[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", sheet[16:24]) == (960, 356 * image_count)
    calls: list[str] = []

    def describe(path: Path, key: str) -> str:
        assert key == "test-key"
        calls.append(path.name)
        return f"Test caption for {path.name}"

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(captions, "describe", describe)
    captions.describe_images(images, edges, montage, output)
    records = json.loads((output / "descriptions.json").read_text())
    assert len(calls) == len(records) == 2 * image_count + 1
    assert {(record["source"], record["image"]) for record in records} == {
        *(("images", name) for name in fixtures),
        *(("edges", Path(name).stem + ".png") for name in fixtures),
        ("montage", "contact-sheet.png"),
    }


def test_missing_edge_does_not_produce_a_montage(tmp_path: Path) -> None:
    images, edges, montage = (tmp_path / name for name in ("images", "edges", "montage"))
    images.mkdir()
    edges.mkdir()
    shutil.copyfile(DEMO / "fixtures/sutro-tower.jpg", images / "sutro-tower.jpg")
    result = run_transform("build-montage", {"images": images, "edges": edges, "montage": montage})
    assert result.returncode != 0
    assert "Could not read" in result.stderr
    assert not (montage / "contact-sheet.png").exists()
