# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Harbor validation contracts shared by submission and SDK discovery."""

from pathlib import Path

import jsonschema
import pytest
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, ResolvedHarborTaskDefinition
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec
from nemo_evaluator.jobs.harbor_scoring import harbor_scoring_task
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_archive import normalize_harbor_instruction
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks


@pytest.mark.parametrize("instruction", ["", " \n\t", "<!-- SPDX-License-Identifier: Apache-2.0 -->\n"])
def test_blank_instructions_fail_discovery_and_snapshot_scoring(tmp_path: Path, instruction: str) -> None:
    """Given blank root instructions, both adapters raise without archive access."""
    root = tmp_path / "task"
    root.mkdir()
    (root / "task.toml").write_text('[task]\nname = "suite/task"\n')
    (root / "instruction.md").write_text(instruction)
    (root / "environment").mkdir()
    (root / "environment/Dockerfile").write_text("FROM ubuntu")
    (root / "tests").mkdir()
    (root / "tests/test.sh").write_text("exit 0")
    definition = ResolvedHarborTaskDefinition.model_validate(
        {
            "kind": "harbor",
            "provenance": {"entity_name": "default/task", "revision_digest": "c" * 64},
            "native_task_id": "task",
            "instruction": instruction,
            "source": {"fileset_ref": "default/files#archive", "files_hash": "a" * 64},
            "harbor_hash": {"digest": "b" * 64, "harbor_version": "0.20.0"},
        }
    )
    with pytest.raises(ValueError, match="task.*instruction is empty"):
        discover_harbor_tasks(tmp_path)
    with pytest.raises(ValueError, match="task.*instruction is empty"):
        harbor_scoring_task(definition, reward_key="reward")


@pytest.mark.parametrize(
    "instruction,expected",
    [
        (None, "task"),
        (" Do it \n", "Do it"),
        ("<!-- SPDX-License-Identifier: Apache-2.0 -->\nDo it", "Do it"),
        ("<!-- ordinary comment -->", "<!-- ordinary comment -->"),
    ],
)
def test_instruction_normalization_preserves_supported_content(instruction: str | None, expected: str) -> None:
    """Given supported instruction content, return its existing normalized representation."""
    assert normalize_harbor_instruction(instruction, task_id="task") == expected


@pytest.mark.parametrize("path", ["a//b", "a/../b", "a/./b", "/a", "a/", "a./b", "a" * 4097])
def test_archive_reference_rejects_noncanonical_paths(path: str) -> None:
    """Given an ambiguous or oversized path, reject the archive reference."""
    with pytest.raises(ValueError):
        HarborArchiveSource(fileset_ref=f"default/files#{path}", files_hash="a" * 64)


def test_archive_schema_documents_server_constraints() -> None:
    """Return a schema that documents constraints not expressible by its regex."""
    field = HarborArchiveSource.model_json_schema()["properties"]["fileset_ref"]
    for rule in ("NFC", "4096 UTF-8 bytes", "a//b", "a/../b", "server"):
        assert rule in field["description"]
    HarborArchiveSource(fileset_ref=field["examples"][0], files_hash="a" * 64)


def test_submit_schema_rejects_empty_arrays_and_preserves_taskset_branch() -> None:
    """Both list branches reject empty tasks; taskset references remain valid selectors."""
    schema = AgentEvalInputSpec.model_json_schema()
    branches = schema["properties"]["tasks"]["anyOf"]
    arrays = [branch for branch in branches if branch.get("type") == "array"]
    assert len(arrays) == 2
    assert all(branch["minItems"] == 1 for branch in arrays)
    assert all("minItems" not in branch for branch in branches if "$ref" in branch)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"tasks": [], "trials": []}, schema)
    jsonschema.validate({"tasks": "default/suite", "trials": []}, schema)
