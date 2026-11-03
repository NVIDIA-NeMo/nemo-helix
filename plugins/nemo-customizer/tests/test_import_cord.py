# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "nemo_customizer"
    / "skills"
    / "nemo-customizer"
    / "scripts"
    / "import_cord.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("import_cord", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["import_cord"] = module
    spec.loader.exec_module(module)
    return module


cord = _load_script()


def test_json2token_wraps_sorted_keys() -> None:
    assert cord.json2token({"menu": {"nm": "BURGER"}}) == "<s_menu><s_nm>BURGER</s_nm></s_menu>"


def test_cord_label_uses_gt_parse_and_the_first_gt_parses_entry() -> None:
    one = json.dumps({"gt_parse": {"nm": "A"}})
    many = json.dumps({"gt_parses": [{"nm": "A"}, {"nm": "B"}]})
    assert cord.cord_label(one) == "<s_nm>A</s_nm>"
    assert cord.cord_label(many) == "<s_nm>A</s_nm>"


def test_cord_label_rejects_a_missing_parse() -> None:
    with pytest.raises(ValueError, match="gt_parse"):
        cord.cord_label(json.dumps({"other": 1}))


class _Image:
    def __init__(self) -> None:
        self.saved: Path | None = None

    def save(self, path: str | Path) -> None:
        self.saved = Path(path)
        self.saved.write_bytes(b"png")


def test_write_split_saves_the_image_beside_the_jsonl(tmp_path: Path) -> None:
    image = _Image()
    count = cord.write_split(
        [{"image": image, "ground_truth": json.dumps({"gt_parse": {"nm": "A"}})}],
        tmp_path,
        "validation.jsonl",
    )
    row = json.loads((tmp_path / "validation.jsonl").read_text())
    assert count == 1
    assert image.saved == tmp_path / "images" / "00000.png"
    assert row == cord.cord_row("images/00000.png", "<s_nm>A</s_nm>")
    assert row["messages"][0]["content"][1]["text"] == cord.CORD_PROMPT
