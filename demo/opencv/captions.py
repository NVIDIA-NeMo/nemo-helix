# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe each input and generated image using NVIDIA's hosted Nemotron 3.5 Super VL model."""

import argparse
import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2  # ty: ignore[unresolved-import]  # Installed in the demo image.

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def describe(path: Path, key: str) -> str:
    photo = cv2.imread(str(path))
    if photo is None:
        raise ValueError(f"Cannot decode {path.name}")
    height, width = photo.shape[:2]
    scale = min(1, 768 / max(height, width))
    preview = cv2.resize(
        photo, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )
    ok, jpeg = cv2.imencode(".jpg", preview)
    if not ok:
        raise ValueError(f"Cannot encode {path.name}")
    encoded = base64.b64encode(jpeg.tobytes()).decode("ascii")
    content = [
        {"type": "text", "text": "Describe this image in one concise sentence using only visible details."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
    ]
    request = Request(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps(
            {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 1024}
        ).encode(),
    )
    try:
        with urlopen(request, timeout=120) as response:
            caption = json.load(response)["choices"][0]["message"]["content"]
    except HTTPError as exc:
        raise SystemExit(f"NVIDIA caption request for {path.name} failed (HTTP {exc.code}).") from exc
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"Model returned an empty caption for {path.name}")
    return caption.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        parser.error("Set NVIDIA_API_KEY for the hosted vision model.")
    records = []
    for source, directory in (("images", args.images), ("edges", args.edges), ("montage", args.montage)):
        paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not paths:
            parser.error(f"No images found in {directory}")
        for path in paths:
            records.append({"source": source, "image": path.name, "model": MODEL, "caption": describe(path, key)})
    args.captions.mkdir(parents=True, exist_ok=True)
    (args.captions / "descriptions.json").write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        print(f"{record['source']}/{record['image']}: {record['caption']}")


if __name__ == "__main__":
    main()
