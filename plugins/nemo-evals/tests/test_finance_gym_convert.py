# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The finance Gym example's converter turns the cookbook's test rows into rows Gym and the scorer accept."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from nhx_evals_sdk.agent_eval.runtimes.gym import discover_gym_tasks

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/finance_gym"
SERVER = EXAMPLE / "environment/resources_servers/finance_qa"


def _load(name: str, path: Path) -> ModuleType:
    # No bytecode: a `__pycache__` beside the example would be uploaded with the environment.
    sys.path.insert(0, str(path.parent))
    writes_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))
        sys.dont_write_bytecode = writes_bytecode


convert_rows = _load("finance_gym_convert_rows", EXAMPLE / "convert_rows.py")
scoring = _load("finance_gym_scoring", SERVER / "scoring.py")

PROMPT = (
    "Task: {task}\nReturn a grounded reasoning trace inside <think>...</think>, followed by only the final answer. "
    "Preserve the requested financial unit or scale.\nQuestion: What is it?"
)


def _row(task: str, answer: str = "42", image: str | None = None) -> dict[str, Any]:
    text = PROMPT.format(task=task)
    if image is None:
        user: Any = text
    else:
        user = [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": text}]
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_an_ocr_row_references_its_image_in_the_fileset() -> None:
    row = convert_rows.gym_row(
        _row("TAT-DQA/multi-span", "2018 | 2019", image="images/page_1.png"), 0, "default/finance-images", "t:1"
    )

    params = row["responses_create_params"]
    assert params["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "fileset://default/finance-images#images/page_1.png",
                    "detail": "auto",
                },
                {"type": "input_text", "text": PROMPT.format(task="TAT-DQA/multi-span")},
            ],
        }
    ]
    assert (params["temperature"], params["max_output_tokens"]) == (0, 512)
    assert {key: row[key] for key in ("expected_answer", "task", "capability", "row_id")} == {
        "expected_answer": "2018 | 2019",
        "task": "TAT-DQA/multi-span",
        "capability": "visual_ocr",
        "row_id": "visual_ocr/0",
    }


@pytest.mark.parametrize(
    ("task", "capability"),
    [
        ("TAT-DQA/span", "visual_ocr"),
        ("TAT-DQA/arithmetic", "reasoning"),
        ("TAT-DQA/count", "reasoning"),
        ("CodeFinQA", "code_generation"),
        ("CodeTAT-QA", "code_generation"),
    ],
)
def test_each_task_gets_its_capability_and_the_scorers_token_limit(task: str, capability: str) -> None:
    row = convert_rows.gym_row(_row(task), 3, None, "t:1")

    assert row["capability"] == capability
    assert row["responses_create_params"]["max_output_tokens"] == scoring.token_limit(task)
    # A text-only prompt stays a plain string.
    assert row["responses_create_params"]["input"] == [{"role": "user", "content": PROMPT.format(task=task)}]


def test_an_image_that_is_already_a_url_is_kept() -> None:
    row = convert_rows.gym_row(_row("TAT-DQA/span", image="fileset://ws/other#a.png"), 0, None, "t:1")

    assert row["responses_create_params"]["input"][0]["content"][0]["image_url"] == "fileset://ws/other#a.png"


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_row("TAT-DQA/span", image="images/a.png"), "pass --image-fileset"),
        (_row("TAT-DQA/unknown"), "unknown task"),
        ({"messages": [{"role": "user", "content": "Question only"}, {"role": "assistant", "content": "1"}]}, "Task:"),
        ({"messages": [{"role": "user", "content": PROMPT.format(task="CodeFinQA")}]}, "reference answer"),
    ],
)
def test_a_row_the_scorer_cannot_use_is_refused(record: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        convert_rows.gym_row(record, 0, None, "test.jsonl:7")


def test_the_command_writes_rows_gym_turns_into_tasks(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "test.jsonl",
        [_row("TAT-DQA/span", "2019", image="images/a.png"), _row("TAT-DQA/multi-span", "a | b", image="images/b.png")],
    )
    output = tmp_path / "visual_ocr.jsonl"

    convert_rows.main([str(source), "--image-fileset", "default/finance-images", "--output", str(output)])

    tasks = discover_gym_tasks(output)
    assert len(tasks) == 2
    extras = [task.metadata["gym_row_extras"] for task in tasks]
    assert sorted(extra["row_id"] for extra in extras) == ["visual_ocr/0", "visual_ocr/1"]
    assert {extra["task"] for extra in extras} == {"TAT-DQA/span", "TAT-DQA/multi-span"}
    assert all(extra["capability"] == "visual_ocr" for extra in extras)


def test_the_command_names_the_bad_line(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    source.write_text(json.dumps(_row("TAT-DQA/span")) + "\n{not json\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"test\.jsonl:2"):
        convert_rows.main([str(source), "--output", str(tmp_path / "out.jsonl")])


def test_only_the_images_the_rows_use_are_copied(tmp_path: Path) -> None:
    images = tmp_path / "data/images"
    images.mkdir(parents=True)
    for name in ("a.png", "b.png", "train-only.png"):
        (images / name).write_bytes(name.encode())
    source = _write(
        tmp_path / "data/test.jsonl",
        [_row("TAT-DQA/span", image="images/a.png"), _row("TAT-DQA/span", "x", image="images/b.png")],
    )
    upload = tmp_path / "upload"

    convert_rows.main(
        [
            str(source),
            "--image-fileset",
            "default/imgs",
            "--copy-images-to",
            str(upload),
            "--output",
            str(tmp_path / "o"),
        ]
    )

    copied = sorted(str(path.relative_to(upload)) for path in upload.rglob("*") if path.is_file())
    assert copied == ["images/a.png", "images/b.png"]
    assert (upload / "images/a.png").read_bytes() == b"a.png"


def test_a_missing_image_stops_the_conversion(tmp_path: Path) -> None:
    source = _write(tmp_path / "test.jsonl", [_row("TAT-DQA/span", image="images/gone.png")])

    with pytest.raises(SystemExit, match="gone.png: an image the rows use is missing"):
        convert_rows.main(
            [
                str(source),
                "--image-fileset",
                "d/i",
                "--copy-images-to",
                str(tmp_path / "up"),
                "--output",
                str(tmp_path / "o"),
            ]
        )
