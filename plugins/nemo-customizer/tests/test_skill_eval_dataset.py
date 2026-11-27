# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for the authored task set, run by the required plugin unit suite.

These validate evaluation inputs without model calls. Agent quality and measured
uplift remain the responsibility of the separately requested live evaluation.
"""

import json

import pytest
from nemo_customizer.skills import get_skills_path

DATASET = json.loads((get_skills_path() / "nemo-customizer" / "evals" / "evals.json").read_text())
CASES = DATASET["evals"]


def test_dataset_identity_and_coverage() -> None:
    assert DATASET["skill_name"] == "nemo-customizer"
    assert len(CASES) >= 10
    assert any(case["should_trigger"] is True for case in CASES)
    assert any(case["should_trigger"] is False for case in CASES)
    ids = [case["id"] for case in CASES]
    assert all(isinstance(case_id, str) and case_id.strip() for case_id in ids)
    assert len(ids) == len(set(ids)), "Duplicate IDs make trial attribution ambiguous"


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_case_has_judge_inputs_and_explicit_routing(case: dict) -> None:
    for field in ("prompt", "expected_output"):
        assert isinstance(case[field], str) and case[field].strip(), field
    assert type(case["should_trigger"]) is bool
    assert case["expected_skill"] == ("nemo-customizer" if case["should_trigger"] else None)
    assert isinstance(case["assertions"], list) and case["assertions"]
    assert all(isinstance(assertion, str) and assertion.strip() for assertion in case["assertions"])
    script = case.get("expected_script")
    if script is not None:
        skill_dir = get_skills_path() / "nemo-customizer"
        path = (skill_dir / script).resolve()
        assert path.is_relative_to(skill_dir.resolve())
        assert path.is_file(), f"Expected script does not exist: {script}"
