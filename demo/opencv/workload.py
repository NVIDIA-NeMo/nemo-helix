# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone equivalents of the two Helix Job steps."""

import argparse
from pathlib import Path

import cv2  # ty: ignore[unresolved-import]  # Installed in the demo image.
import numpy as np


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["detect-edges", "build-montage"])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--montage", type=Path)
    args = parser.parse_args()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        parser.error("The input directory needs at least one photo")
    if len({p.stem for p in paths}) != len(paths):
        parser.error("Give each input photo a unique filename stem")
    if args.step == "build-montage" and args.montage is None:
        parser.error("build-montage requires --montage")

    rows = []
    for path in paths:
        output = args.edges / (path.stem + ".png")
        original = read(path, cv2.IMREAD_COLOR)
        if args.step == "detect-edges":
            edge = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 100, 200)
            write(output, edge)
            print(output.name)
        else:
            edge = read(output, cv2.IMREAD_COLOR)
            if edge.shape != original.shape:
                raise ValueError(f"Edge image dimensions differ for {path.name}")
            label = np.full((36, 960, 3), 28, dtype=np.uint8)
            for x, title in [(10, path.name), (490, "Canny edges")]:
                cv2.putText(label, title, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
            rows.extend([label, np.hstack([panel(original), panel(edge)])])
    if rows:
        write(args.montage / "contact-sheet.png", np.vstack(rows))
        print(f"contact-sheet.png: {len(paths)} row(s)")


if __name__ == "__main__":
    main()
