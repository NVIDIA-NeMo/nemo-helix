# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path

import pytest
from nhx.automodel.tasks.training.backends import vlm
from nhx.automodel.tasks.training.backends.vlm import rewrite_fileset_rows
from nhx.automodel.tasks.training.datasets.preparation import DatasetFormatError


def _rewrite(tmp_path: Path, url: str) -> dict:
    dataset = tmp_path / "data"
    dataset.mkdir(exist_ok=True)
    source = dataset / "training.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"
    rewrite_fileset_rows(source, dest, dataset_root=dataset, image_dir=tmp_path / "images", column_keys=None)
    return json.loads(dest.read_text(encoding="utf-8"))


def test_a_relative_image_url_stays_under_the_dataset(tmp_path: Path) -> None:
    row = _rewrite(tmp_path, "scans/receipt.png")

    assert row["conversation"][0]["content"][0] == {
        "type": "image",
        "image": str(tmp_path / "data" / "scans" / "receipt.png"),
    }


def test_thinking_tool_calls_and_tools_survive_the_rewrite(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    dataset.mkdir()
    source = dataset / "training.jsonl"
    tool_call = {"type": "function", "function": {"name": "lookup", "arguments": {"q": "a"}}}
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "description": "look", "parameters": {"type": "object", "properties": {}}},
        }
    ]
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "where"},
                    {"role": "assistant", "thinking": "check the map"},
                    {"role": "assistant", "tool_calls": [tool_call]},
                    {"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}, "listen"]},
                ],
                "tools": tools,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"
    rewrite_fileset_rows(source, dest, dataset_root=dataset, image_dir=tmp_path / "images", column_keys=None)
    row = json.loads(dest.read_text(encoding="utf-8"))

    assert row["conversation"][1] == {"role": "assistant", "thinking": "check the map"}
    assert row["conversation"][2] == {"role": "assistant", "tool_calls": [tool_call]}
    assert row["conversation"][3]["content"] == [
        {"type": "audio", "audio": "clip.wav"},
        {"type": "text", "text": "listen"},
    ]
    assert row["tools"] == tools

    loaded = vlm.make_conversation_dataset(str(dest))
    assert loaded[0]["tools"] == tools
    assert loaded[0]["conversation"][1]["thinking"] == "check the map"
    assert loaded[0]["conversation"][2]["tool_calls"] == [tool_call]


def test_an_absolute_image_url_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetFormatError, match="relative to the dataset"):
        _rewrite(tmp_path, "/etc/shadow")


def test_an_image_url_cannot_leave_the_dataset(tmp_path: Path) -> None:
    with pytest.raises(DatasetFormatError, match="outside the dataset"):
        _rewrite(tmp_path, "../../etc/shadow")


def test_an_image_url_cannot_escape_through_a_symlink(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "outside").symlink_to(tmp_path.parent, target_is_directory=True)

    with pytest.raises(DatasetFormatError, match="outside the dataset"):
        _rewrite(tmp_path, "outside/secret.png")


def test_inline_images_require_an_allowed_image_media_type(tmp_path: Path) -> None:
    payload = base64.b64encode(b"not an image").decode("ascii")

    with pytest.raises(DatasetFormatError, match="image/png"):
        _rewrite(tmp_path, f"data:text/plain;base64,{payload}")


def test_inline_images_require_valid_base64(tmp_path: Path) -> None:
    with pytest.raises(DatasetFormatError, match="invalid base64"):
        _rewrite(tmp_path, "data:image/png;base64,%%%")


def test_inline_images_are_size_limited_before_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vlm, "MAX_INLINE_IMAGE_BYTES", 2)
    payload = base64.b64encode(b"123").decode("ascii")

    with pytest.raises(DatasetFormatError, match="size limit"):
        _rewrite(tmp_path, f"data:image/png;base64,{payload}")
