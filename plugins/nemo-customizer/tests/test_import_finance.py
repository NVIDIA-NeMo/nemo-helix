# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "nemo_customizer"
    / "skills"
    / "nemo-customizer"
    / "scripts"
    / "import_finance.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("import_finance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["import_finance"] = module
    spec.loader.exec_module(module)
    return module


finance = _load_script()


def _row(**overrides: object) -> dict:
    row = {
        "sample_id": "tatdqa:q",
        "source_dataset": "TAT-DQA",
        "task": "TAT-DQA/span",
        "task_family": "visual_document_reasoning",
        "lineage_key": "tatdqa-report:acme",
        "question_key": "what is revenue",
        "image_path": "train/q.png",
        "context": "revenue 10",
        "question": "What is revenue?",
        "reasoning": "Read the page.",
        "answer": "10",
        "program": "",
        "target": "<think>\nRead the page.\n</think>\n10",
        "assigned_split": "train",
    }
    row.update(overrides)
    return row


def test_ocr_row_uses_an_image_part_and_the_thinking_target() -> None:
    projected = finance.project_row(_row(), "visual_ocr", "train")
    message = finance.finance_row(projected)
    user = message["messages"][0]["content"]
    assert user[0] == {"type": "image_url", "image_url": {"url": "train/q.png"}}
    assert user[1]["type"] == "text"
    assert "What is revenue?" in user[1]["text"]
    assert message["messages"][1]["content"] == "<think>\nRead the page.\n</think>\n10"


def test_reasoning_row_is_text_and_drops_the_image() -> None:
    projected = finance.project_row(_row(task="TAT-DQA/arithmetic"), "reasoning", "train")
    message = finance.finance_row(projected)
    assert projected["image_path"] == ""
    assert isinstance(message["messages"][0]["content"], str)
    assert "Evidence:" in message["messages"][0]["content"]
    assert "image_url" not in json.dumps(message)


def test_test_row_assistant_is_the_answer_only() -> None:
    projected = finance.project_row(_row(task="CodeFinQA", program="print(1)"), "code_generation", "test")
    message = finance.finance_row(projected)
    assert projected["reasoning"] == ""
    assert projected["program"] == ""
    assert message["messages"][1]["content"] == "10"
    assert set(message) == {"messages"}


def test_write_capability_records_the_leakage_audit(tmp_path: Path) -> None:
    media = tmp_path / "pages"
    image = media / "train" / "q.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    splits = {
        "train": [_row(sample_id="train", question_key="train q", lineage_key="report:train")],
        "validation": [_row(sample_id="val", question_key="val q", lineage_key="report:val", image_path="train/q.png")],
        "test": [_row(sample_id="test", question_key="test q", lineage_key="report:test", image_path="train/q.png")],
    }
    manifest = finance.write_capability(
        tmp_path / "visual_ocr",
        splits,
        "visual_ocr",
        media,
        seed=42,
        validation_ratio=0.1,
        render_dpi=150,
        parent_audit={"overlaps": {}},
    )
    written = json.loads((tmp_path / "visual_ocr" / "test.jsonl").read_text().splitlines()[0])
    assert written["messages"][1]["content"] == "10"
    assert (tmp_path / "visual_ocr" / "images" / "q.png").is_file()
    assert manifest["overlaps"]["train__test"]["document_lineages"] == 0
    assert "parent_audit" in json.loads((tmp_path / "visual_ocr" / "split_manifest.json").read_text())


def test_make_splits_keeps_the_official_bizbench_test_and_blocks_leakage() -> None:
    tat = [
        _row(
            sample_id=f"tat:{index}",
            question_key=f"question {index}",
            lineage_key=f"tatdqa-report:{index}",
            question=f"Question {index}?",
        )
        for index in range(180)
    ]
    official = _row(
        sample_id="biz:test",
        source_dataset="BizBench",
        task="CodeFinQA",
        question_key="held out",
        lineage_key="bizbench-context:held",
        question="Held out?",
        assigned_split="test",
        program="print(1)",
        reasoning="print(1)",
    )
    colliding = _row(
        sample_id="biz:train",
        source_dataset="BizBench",
        task="CodeFinQA",
        question_key="held out",
        lineage_key="bizbench-context:other",
        question="Held out?",
        assigned_split="train",
        program="print(2)",
        reasoning="print(2)",
    )
    splits, audit = finance.make_splits(tat, [official, colliding], seed=42, validation_ratio=0.1)
    assert any(row["sample_id"] == "biz:test" for row in splits["test"])
    assert all(row["sample_id"] != "biz:train" for split in splits.values() for row in split)
    assert audit["overlaps"]["train__test"]["normalized_questions"] == 0
