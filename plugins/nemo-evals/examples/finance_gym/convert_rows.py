#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the finance cookbook's test rows into NeMo Gym rows for the ``finance_qa`` environment.

The input is one capability's chat-transcript JSONL: a user turn holding the prompt, whose first line is
``Task: <task>``, then an assistant turn holding the reference answer. OCR rows put the page image first, as a
path relative to the dataset.

Each Gym row keeps the prompt unchanged and adds the cookbook's sampling settings: temperature 0, and 256 new
tokens for arithmetic and count or 512 for every other task. It carries ``expected_answer`` and ``task`` for the
scorer, and ``capability`` and ``row_id`` for the report. An image path becomes a ``fileset://`` reference into
the FileSet the images were uploaded to, and the Inference Gateway loads it when the model is called.
``--copy-images-to`` copies just the images the rows use, at their relative paths, into a directory to upload as
that FileSet: the dataset's image directory also holds the training splits' images.

    python3 convert_rows.py visual_ocr/test.jsonl --image-fileset default/finance-images \\
      --copy-images-to ocr-images --output visual_ocr.jsonl
    python3 convert_rows.py reasoning/test.jsonl --output reasoning.jsonl
    python3 convert_rows.py code_generation/test.jsonl --output code_generation.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any

#: The cookbook's tasks, by the capability whose adapter answers them.
TASK_CAPABILITIES = {
    "TAT-DQA/span": "visual_ocr",
    "TAT-DQA/multi-span": "visual_ocr",
    "TAT-DQA/arithmetic": "reasoning",
    "TAT-DQA/count": "reasoning",
    "CodeFinQA": "code_generation",
    "CodeTAT-QA": "code_generation",
}
#: Tasks the cookbook gives 256 new tokens. Every other task gets 512.
SHORT_ANSWER_TASKS = frozenset({"TAT-DQA/arithmetic", "TAT-DQA/count"})
_URL_SCHEMES = ("fileset://", "http://", "https://", "data:")


def token_limit(task: str) -> int:
    """The cookbook's new-token limit for ``task``."""
    return 256 if task in SHORT_ANSWER_TASKS else 512


def _dataset_path(path: str, where: str) -> str:
    """``path``, an image path relative to the dataset, refused if it is absolute or climbs out with ``..``."""
    parts = PurePosixPath(path)
    if parts.is_absolute() or ".." in parts.parts:
        raise ValueError(f"{where}: an image path must stay inside the dataset ({path})")
    return path


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text") or "" for part in content if isinstance(part, dict) and part.get("type") == "text")


def _input_content(content: Any, image_fileset: str | None, where: str) -> str | list[dict[str, Any]]:
    """A chat message's content in the Responses API's input form."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": "input_text", "text": part["text"]})
        elif kind == "image_url":
            url = part["image_url"]["url"]
            if not url.startswith(_URL_SCHEMES):
                if image_fileset is None:
                    raise ValueError(f"{where}: an image is a relative path ({url}); pass --image-fileset")
                url = f"fileset://{image_fileset}#{_dataset_path(url, where)}"
            # Gym's Responses types require `detail` on an image part.
            parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
        else:
            raise ValueError(f"{where}: unsupported content part type {kind!r}")
    return parts


def gym_row(row: dict[str, Any], index: int, image_fileset: str | None, where: str) -> dict[str, Any]:
    """One chat-transcript row as a Gym row."""
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError(f"{where}: expected messages ending with the assistant's reference answer")
    prompt = messages[:-1]
    user_text = _text(next((m["content"] for m in prompt if m.get("role") == "user"), ""))
    first_line = user_text.split("\n", 1)[0]
    if not first_line.startswith("Task: "):
        raise ValueError(f"{where}: the prompt's first line is not 'Task: <task>'")
    task = first_line.removeprefix("Task: ").strip()
    if task not in TASK_CAPABILITIES:
        raise ValueError(f"{where}: unknown task {task!r}")
    capability = TASK_CAPABILITIES[task]
    return {
        "responses_create_params": {
            "input": [
                {"role": message["role"], "content": _input_content(message["content"], image_fileset, where)}
                for message in prompt
            ],
            "temperature": 0,
            "max_output_tokens": token_limit(task),
        },
        "expected_answer": _text(messages[-1]["content"]).strip(),
        "task": task,
        "capability": capability,
        "row_id": f"{capability}/{index}",
    }


def image_paths(record: dict[str, Any]) -> list[str]:
    """The relative image paths a chat-transcript row uses."""
    return [
        part["image_url"]["url"]
        for message in record.get("messages") or []
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url" and not part["image_url"]["url"].startswith(_URL_SCHEMES)
    ]


def convert(source: Path, image_fileset: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Every row of ``source``, converted, and the relative image paths they use."""
    rows: list[dict[str, Any]] = []
    images: dict[str, None] = {}
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            where = f"{source}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: not a JSON record: {error}") from error
            rows.append(gym_row(record, len(rows), image_fileset, where))
            images.update(dict.fromkeys(image_paths(record)))
    return rows, list(images)


def copy_images(source_root: Path, images: list[str], destination: Path) -> None:
    """Copy each image from ``source_root`` to ``destination``, keeping its relative path.

    An image is refused if its path, or a symlink on the way, leads outside either directory: a dataset could
    otherwise put files outside ``destination``, or copy files from elsewhere into what is uploaded.
    """
    source_dir, destination_dir = source_root.resolve(), destination.resolve()
    for relative in images:
        _dataset_path(relative, str(source_root))
        found = source_root / relative
        target = destination / relative
        if not found.resolve().is_relative_to(source_dir):
            raise ValueError(f"{found}: links outside the dataset")
        if not target.resolve().is_relative_to(destination_dir):
            raise ValueError(f"{target}: links outside {destination}")
        if not found.is_file():
            raise ValueError(f"{found}: an image the rows use is missing")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, target)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="One capability's chat-transcript JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the Gym rows")
    parser.add_argument(
        "--image-fileset",
        metavar="WORKSPACE/NAME",
        help="The FileSet the images were uploaded to, keeping their relative paths; required for OCR rows",
    )
    parser.add_argument(
        "--copy-images-to",
        type=Path,
        metavar="DIR",
        help="Copy the images the rows use into DIR, at their relative paths, to upload as that FileSet",
    )
    args = parser.parse_args(argv)
    source = args.source.expanduser()
    try:
        rows, images = convert(source, args.image_fileset)
        if args.copy_images_to is not None:
            copy_images(source.parent, images, args.copy_images_to.expanduser())
    except ValueError as error:
        sys.exit(str(error))
    with args.output.expanduser().open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tasks = sorted({row["task"] for row in rows})
    print(f"{len(rows)} rows ({', '.join(tasks)}) -> {args.output}", file=sys.stderr)
    if args.copy_images_to is not None:
        print(f"{len(images)} images -> {args.copy_images_to}", file=sys.stderr)


if __name__ == "__main__":
    main()
