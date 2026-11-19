# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transform images staged by the Jobs launcher; the controller publishes output."""

import argparse
from pathlib import Path

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
        print(f"[detect-edges] Detecting edges: {path.name}", flush=True)
        gray = cv2.cvtColor(read(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        write(edges / (path.stem + ".png"), cv2.Canny(gray, 100, 200))


def build_montage(images: Path, edges: Path, montage: Path):
    rows = []
    for path in photos(images):
        print(f"[build-montage] Adding row: {path.name}", flush=True)
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
    parser = argparse.ArgumentParser(description=__doc__)
    transforms = parser.add_subparsers(dest="transform", required=True)
    detect = transforms.add_parser("detect-edges")
    detect.add_argument("--images", type=Path, required=True)
    detect.add_argument("--edges", type=Path, required=True)
    montage = transforms.add_parser("build-montage")
    montage.add_argument("--images", type=Path, required=True)
    montage.add_argument("--edges", type=Path, required=True)
    montage.add_argument("--montage", type=Path, required=True)
    describe = transforms.add_parser("describe-images")
    describe.add_argument("--images", type=Path, required=True)
    describe.add_argument("--edges", type=Path, required=True)
    describe.add_argument("--montage", type=Path, required=True)
    describe.add_argument("--captions", type=Path, required=True)
    args = parser.parse_args()
    print(f"[{args.transform}] Inputs staged; starting transform", flush=True)
    if args.transform == "detect-edges":
        output = args.edges
        output.mkdir(parents=True, exist_ok=True)
        detect_edges(args.images, args.edges)
    elif args.transform == "build-montage":
        output = args.montage
        output.mkdir(parents=True, exist_ok=True)
        build_montage(args.images, args.edges, args.montage)
    else:
        output = args.captions
        output.mkdir(parents=True, exist_ok=True)
        describe_images(args.images, args.edges, args.montage, args.captions)
    artifacts = sorted(output.iterdir())
    if not artifacts:
        raise RuntimeError(f"{args.transform} produced no artifacts")
    print(f"[{args.transform}] Produced {len(artifacts)} artifact(s); controller will upload and commit", flush=True)


if __name__ == "__main__":
    main()
