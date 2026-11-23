# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage this demo's flat FileSets, execute the transform, and publish its output."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2  # ty: ignore[unresolved-import]  # Installed in the demo image.
import numpy as np
from captions import describe_images


def read(path, mode):
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Could not read {path}")
    return image


def write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def panel(image):
    height, width = image.shape[:2]
    scale = min(480 / width, 320 / height)
    size = max(1, round(width * scale)), max(1, round(height * scale))
    thumb = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    canvas = np.full((320, 480, 3), 28, dtype=np.uint8)
    x, y = (480 - size[0]) // 2, (320 - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = thumb
    return canvas


def photos(images: Path) -> list[Path]:
    paths = sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("The input FileSet needs photos with distinct filename stems")
    return paths


def detect_edges(images: Path, edges: Path):
    for path in photos(images):
        gray = cv2.cvtColor(read(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        write(edges / (path.stem + ".png"), cv2.Canny(gray, 100, 200))


def build_montage(images: Path, edges: Path, montage: Path):
    rows = []
    for path in photos(images):
        original = read(path, cv2.IMREAD_COLOR)
        edge = read(edges / (path.stem + ".png"), cv2.IMREAD_COLOR)
        if edge.shape != original.shape:
            raise ValueError(f"Edge image dimensions differ for {path.name}")
        label = np.full((36, 960, 3), 28, dtype=np.uint8)
        for x, title in [(10, path.name), (490, "Canny edges")]:
            cv2.putText(label, title, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
        rows.extend([label, np.hstack([panel(original), panel(edge)])])
    write(montage / "contact-sheet.png", np.vstack(rows))


def main():
    config = json.loads(Path(os.environ["NEMO_JOB_STEP_CONFIG_FILE_PATH"]).read_text())
    workspace = quote(os.environ["NEMO_JOB_WORKSPACE"], safe="")
    base = os.environ["NHX_BASE_URL"].rstrip("/") + f"/apis/files/v2/workspaces/{workspace}/filesets"
    with TemporaryDirectory() as temp:
        directories = {}
        for role, name in config["inputs"].items():
            local = Path(temp, role)
            local.mkdir()
            remote = f"{base}/{quote(name, safe='')}"
            with urlopen(remote + "/files", timeout=30) as response:
                entries = json.load(response)["data"]
            for entry in entries:
                path = entry["path"]
                if Path(path).name != path or path in {"", ".", ".."}:
                    raise ValueError(f"Expected a flat demo FileSet, got {path!r}")
                with urlopen(remote + "/-/" + quote(path, safe=""), timeout=30) as response:
                    (local / path).write_bytes(response.read())
            directories[role] = local

        output_role = config["output_role"]
        output = Path(temp, output_role)
        output.mkdir()
        directories[output_role] = output
        step = config["transform"]
        transforms = {"detect-edges": detect_edges, "build-montage": build_montage, "describe-images": describe_images}
        transforms[step](**directories)

        remote = f"{base}/{quote(config['output_name'], safe='')}"
        artifacts = sorted(output.iterdir())
        if not artifacts:
            raise RuntimeError(f"{step} produced no artifacts")
        for artifact in artifacts:
            request = Request(
                remote + "/-/" + quote(artifact.name, safe=""),
                method="PUT",
                headers={"Content-Type": "application/octet-stream"},
                data=artifact.read_bytes(),
            )
            with urlopen(request, timeout=30):
                pass
        with urlopen(Request(remote + "/commit", method="POST"), timeout=30) as response:
            print(f"Published {config['output_name']} at {json.load(response)['commit_revision']}")


if __name__ == "__main__":
    main()
