# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

ROWS = Path(__file__).parents[3] / "scripts" / "grpo-examples" / "gym_rows.py"
SPEC = importlib.util.spec_from_file_location("gym_rows", ROWS)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_image_message_puts_the_image_before_the_text() -> None:
    message = MODULE.image_message("how many cubes?", ["data:image/png;base64,aa"])
    assert message["role"] == "user"
    assert message["content"] == [
        {"type": "input_image", "image_url": "data:image/png;base64,aa", "detail": "auto"},
        {"type": "input_text", "text": "how many cubes?"},
    ]


def test_write_rows_numbers_each_repeat(tmp_path: Path) -> None:
    agent = {"type": "responses_api_agents", "name": "clevr_cogent_simple_agent"}
    written = MODULE.write_rows(tmp_path / "training.jsonl", [{"question": "q"}], agent, repeats=2)
    lines = (tmp_path / "training.jsonl").read_text(encoding="utf-8").splitlines()
    assert written == 2
    assert [json.loads(line)["task_idx"] for line in lines] == [0, 1]
    assert json.loads(lines[1])["agent_ref"] == agent
