# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a fileset into the conversation rows FinetuneRecipeForVLM collates.

Automodel's collate reads ``{"conversation": [{"role", "content": [parts]}]}``
with ``{type: image, image: path}`` and ``{type: text, text}``. A fileset row
is still OpenAI chat (``image_url``) or prompt/completion. The compiler writes
the conversation JSONL; this loader injects the fake image that keeps FSDP
from skipping the vision tower on a text row.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

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
                dest_handle.write(json.dumps({"conversation": conversation}, ensure_ascii=False) + "\n")
        temp_path.replace(dest)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def make_conversation_dataset(path_or_dataset: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Load conversation JSONL and mark text rows for the fake-image mask.

    ``FinetuneRecipeForVLM.build_dataloader`` instantiates this with
    ``path_or_dataset``. The collate masks vision tokens only when the example
    carries ``_injected_fake``, so the image has to be in the conversation here.
    """
    del kwargs
    examples: list[dict[str, Any]] = []
    with Path(path_or_dataset).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            conversation = json.loads(line)["conversation"]
            example: dict[str, Any] = {"conversation": conversation}
            if not _conversation_has_media(conversation):
                example["conversation"] = _inject_fake_image(conversation)
                example["_injected_fake"] = True
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
            conversation.append(
                {
                    "role": message["role"],
                    "content": _content_parts(message.get("content"), dataset_root=dataset_root, image_dir=image_dir),
                }
            )
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
        return [{"type": "text", "text": "" if content is None else str(content)}]
    converted: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            converted.append({"type": "text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image_url":
            url = str((part.get("image_url") or {}).get("url", ""))
            converted.append({"type": "image", "image": _resolve_image(url, dataset_root, image_dir)})
    return converted


def _resolve_image(url: str, dataset_root: Path, image_dir: Path) -> str:
    """Map an image URL onto a file under ``dataset_root``.

    A ``data:`` URL is decoded into ``image_dir``. Any other URL is a path
    relative to the dataset. An absolute path is not joined: ``Path / "/etc/shadow"``
    discards the root and opens that file.
    """
    if url.startswith("data:"):
        return _write_data_url(url, image_dir)
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


def _conversation_has_media(conversation: list[dict[str, Any]]) -> bool:
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "video"}:
                return True
    return False


def _inject_fake_image(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer Automodel's 56x56 image. Tests without that package still set the flag."""
    try:
        from nemo_automodel.components.datasets.vlm.fake_image import inject_fake_image_into_conversation
    except ImportError:
        return _inject_placeholder(conversation)
    return inject_fake_image_into_conversation(conversation)


def _inject_placeholder(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conversation = copy.deepcopy(conversation)
    placeholder = {"type": "image", "image": "fake"}
    for message in conversation:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content.insert(0, placeholder)
        else:
            message["content"] = [placeholder, {"type": "text", "text": content or ""}]
        return conversation
    conversation.insert(0, {"role": "user", "content": [placeholder]})
    return conversation
