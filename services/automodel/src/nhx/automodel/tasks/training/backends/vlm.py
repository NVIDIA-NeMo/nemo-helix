# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a fileset into the conversation rows FinetuneRecipeForVLM collates.

Automodel's collate reads ``{"conversation": [{"role", "content": [parts]}]}``
with ``{type: image, image: path}`` and ``{type: text, text}``. A fileset row
is still OpenAI chat (``image_url``) or prompt/completion. The compiler writes
the conversation JSONL and sets ``dataset.max_length``, which turns on
Automodel's pretokenized wrapper. That wrapper injects the fake image that
keeps FSDP from skipping the vision tower on a text row, and masks it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nhx.automodel.tasks.training.datasets.preparation import DatasetFormatError

logger = logging.getLogger(__name__)

CONVERSATION_DATASET_TARGET = "nhx.automodel.tasks.training.backends.vlm.make_conversation_dataset"
MAX_INLINE_IMAGE_BYTES = 25 * 1024 * 1024
_INLINE_IMAGE_EXTENSIONS = {
    "image/gif": "gif",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def rewrite_fileset_rows(
    source: Path,
    dest: Path,
    dataset_root: Path,
    image_dir: Path,
    column_keys: tuple[str, str] | None,
) -> None:
    """Write conversation JSONL. Image bytes land under ``image_dir``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with (
            source.open(encoding="utf-8") as source_handle,
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=dest.parent,
                prefix=f".{dest.name}.",
                delete=False,
            ) as dest_handle,
        ):
            temp_path = Path(dest_handle.name)
            for line in source_handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                conversation = _row_to_conversation(
                    row,
                    dataset_root=dataset_root,
                    image_dir=image_dir,
                    column_keys=column_keys,
                )
                record: dict[str, Any] = {"conversation": conversation}
                if isinstance(row.get("tools"), list):
                    record["tools"] = row["tools"]
                dest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path.replace(dest)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def make_conversation_dataset(path_or_dataset: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Load conversation JSONL written by the compiler.

    ``FinetuneRecipeForVLM`` instantiates this with ``path_or_dataset``.
    ``max_length`` stays on the dataset config for Automodel's pretokenized
    wrapper, which truncates to that length and injects the fake image.
    """
    del kwargs
    examples: list[dict[str, Any]] = []
    with Path(path_or_dataset).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parsed = json.loads(line)
            example: dict[str, Any] = {"conversation": parsed["conversation"]}
            if "tools" in parsed:
                example["tools"] = parsed["tools"]
            examples.append(example)
    return examples


def _row_to_conversation(
    row: dict[str, Any],
    dataset_root: Path,
    image_dir: Path,
    column_keys: tuple[str, str] | None,
) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        conversation = []
        for message in messages:
            converted = {"role": message["role"]}
            content = message.get("content")
            if content is not None:
                converted["content"] = _content_parts(content, dataset_root=dataset_root, image_dir=image_dir)
            if message.get("thinking") is not None:
                converted["thinking"] = message["thinking"]
            if message.get("tool_calls"):
                converted["tool_calls"] = message["tool_calls"]
            conversation.append(converted)
        return conversation

    if column_keys is None:
        column_keys = ("prompt", "completion")
    question, answer = column_keys
    return [
        {"role": "user", "content": [{"type": "text", "text": str(row.get(question, ""))}]},
        {"role": "assistant", "content": [{"type": "text", "text": str(row.get(answer, ""))}]},
    ]


def _content_parts(content: Any, dataset_root: Path, image_dir: Path) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts = [content] if isinstance(content, dict) else content
    if not isinstance(parts, list):
        return [{"type": "text", "text": str(content)}]
    converted: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            converted.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            converted.append({"type": "text", "text": str(part)})
            continue
        if part.get("type") == "text":
            converted.append({"type": "text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image_url":
            url = str((part.get("image_url") or {}).get("url", ""))
            converted.append({"type": "image", "image": _resolve_image(url, dataset_root, image_dir)})
        else:
            converted.append(dict(part))
    return converted


def _resolve_image(url: str, dataset_root: Path, image_dir: Path) -> str:
    """Map an image URL onto a file under ``dataset_root``.

    A ``data:`` URL is decoded into ``image_dir``. Any other URL is a path
    relative to the dataset, and that file must already be in the fileset.
    An ``http`` or ``https`` URL is not a path: ``Path`` does not treat it as
    absolute, so joining it would look for a file named after the URL. An
    absolute path is not joined either: ``Path / "/etc/shadow"`` discards the
    root and opens that file.
    """
    if url.startswith("data:"):
        return _write_data_url(url, image_dir)
    scheme = urlparse(url).scheme
    if scheme:
        raise DatasetFormatError(
            f"Image URL {url!r} must be a path relative to the dataset directory or a data: URL. "
            f"Remote {scheme} URLs are not downloaded."
        )
    relative = Path(url)
    if relative.is_absolute():
        raise DatasetFormatError(f"Image URL {url!r} must be a path relative to the dataset directory.")
    root = dataset_root.resolve()
    try:
        resolved = (root / relative).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise DatasetFormatError(f"Image URL {url!r} is not a valid dataset-relative path.") from exc
    if not resolved.is_relative_to(root):
        raise DatasetFormatError(f"Image URL {url!r} resolves outside the dataset directory.")
    if not resolved.is_file():
        raise DatasetFormatError(f"Image URL {url!r} does not exist in the dataset.")
    return str(resolved)


def _write_data_url(url: str, image_dir: Path) -> str:
    header, _, payload = url.partition(",")
    media_type, *parameters = header[5:].lower().split(";")
    extension = _INLINE_IMAGE_EXTENSIONS.get(media_type)
    if extension is None or "base64" not in parameters:
        raise DatasetFormatError(
            "Inline images must be base64 data URLs with image/png, image/jpeg, image/gif, or image/webp content."
        )
    # Reject before decoding so an oversized string cannot allocate a second,
    # decoded buffer. Base64 expands bytes by at least 4/3.
    if len(payload) > ((MAX_INLINE_IMAGE_BYTES + 2) // 3) * 4:
        raise DatasetFormatError(f"Inline image exceeds the {MAX_INLINE_IMAGE_BYTES}-byte size limit.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DatasetFormatError("Inline image contains invalid base64 data.") from exc
    if len(raw) > MAX_INLINE_IMAGE_BYTES:
        raise DatasetFormatError(f"Inline image exceeds the {MAX_INLINE_IMAGE_BYTES}-byte size limit.")

    image_dir.mkdir(parents=True, exist_ok=True)
    dest = image_dir / f"{hashlib.sha256(raw).hexdigest()[:16]}.{extension}"
    # Replacing a path does not follow a pre-existing symlink at ``dest``.
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=image_dir, prefix=".inline-image.", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(raw)
        temp_path.replace(dest)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return str(dest)
