#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write CORD-v2 train and validation into one fileset directory.

Run this before upload. Automodel does not convert the Hugging Face dataset.
The directory you upload is already the fileset: ``train/`` and ``validation/``,
each with its JSONL and ``images/``.

    uv run --no-project --with datasets --with pillow \
      plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/import_cord.py \
      --output /tmp/cord-v2

Then upload that directory (a new fileset name, no ``--exist-ok``):

    nemo files filesets create cord-v2 --workspace default --purpose dataset
    nemo files upload /tmp/cord-v2 cord-v2 --workspace default

The fileset holds ``train/train.jsonl`` and ``validation/validation.jsonl``.
Automodel selects training files from ``train/`` and validation files from
``validation/``. Job JSON names this one fileset. The Evaluator scores
``validation/validation.jsonl`` in it: that official validation file is the
frozen split, and the script does not write ``test.jsonl``. Do not reshuffle
it between base and tuned runs.

Image urls are fileset-relative (``train/images/00000.png``,
``validation/images/00000.png``).

``json2token`` matches the Automodel CORD loader. When a row has ``gt_parses``,
the first parse is kept so a second run writes the same label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Protocol, cast

CORD_DATASET = "naver-clova-ix/cord-v2"
CORD_PROMPT = "Describe this image."


class _Image(Protocol):
    def save(self, path: str | Path) -> None: ...


def json2token(obj: Any, sort_json_key: bool = True) -> str:
    """Convert an ordered JSON object into the CORD tagged token string."""
    if type(obj) is dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        output = ""
        keys = sorted(cast(Iterable[str], obj.keys()), reverse=True) if sort_json_key else obj.keys()
        for key in keys:
            output += rf"<s_{key}>" + json2token(obj[key], sort_json_key) + rf"</s_{key}>"
        return output
    if type(obj) is list:
        return r"<sep/>".join(json2token(item, sort_json_key) for item in obj)
    return str(obj)


def cord_label(ground_truth: str) -> str:
    """Return the ``json2token`` label for one CORD ``ground_truth`` string."""
    parsed = json.loads(ground_truth)
    if "gt_parses" in parsed:
        parses = parsed["gt_parses"]
        if not isinstance(parses, list) or not parses:
            raise ValueError("gt_parses must be a non-empty list")
        return json2token(parses[0], sort_json_key=True)
    parse = parsed.get("gt_parse")
    if not isinstance(parse, dict):
        raise ValueError("ground_truth must contain gt_parse or gt_parses")
    return json2token(parse, sort_json_key=True)


def cord_row(image_url: str, label: str) -> dict[str, Any]:
    """One multimodal chat row. The assistant value is the label string."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": CORD_PROMPT},
                ],
            },
            {"role": "assistant", "content": label},
        ]
    }


def write_split(examples: Iterable[dict[str, Any]], dest: Path, jsonl_name: str, url_prefix: str = "") -> int:
    """Write ``jsonl_name`` and ``images/`` under ``dest``. Return the row count.

    ``url_prefix`` is the directory name under the fileset root, so the image
    url is fileset-relative (``train/images/00000.png``).
    """
    images = dest / "images"
    images.mkdir(parents=True, exist_ok=True)
    image_dir = f"{url_prefix}/images" if url_prefix else "images"
    count = 0
    with (dest / jsonl_name).open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples):
            relative = f"{image_dir}/{index:05d}.png"
            image: _Image = example["image"]
            image.save(images / f"{index:05d}.png")
            row = cord_row(relative, cord_label(example["ground_truth"]))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_split(split: str, limit: int | None) -> Any:
    """Load one CORD-v2 split. ``limit`` keeps the first N rows, in dataset order."""
    from datasets import load_dataset

    spec = split if limit is None else f"{split}[:{limit}]"
    return load_dataset(CORD_DATASET, split=spec)


def write_filesets(output: Path, limit: int | None = None) -> dict[str, int]:
    """Write ``output/train`` and ``output/validation`` for one fileset upload."""
    counts = {
        "train": write_split(load_split("train", limit), output / "train", "train.jsonl", "train"),
        "validation": write_split(
            load_split("validation", limit), output / "validation", "validation.jsonl", "validation"
        ),
    }
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fileset directory: train/ and validation/")
    parser.add_argument("--limit", type=int, default=None, help="First N rows of each split")
    args = parser.parse_args()
    counts = write_filesets(args.output, args.limit)
    print(f"Wrote {counts['train']} train rows and {counts['validation']} validation rows to {args.output}")


if __name__ == "__main__":
    main()
