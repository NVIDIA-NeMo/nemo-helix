# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from _doubles import make_job_context
from nemo_agent_hardener_plugin.jobs import _common
from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob
from nemo_agent_hardener_plugin.jobs.synth_benign import AgentHardenerSynthBenignJob
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("job_class", "config"),
    [
        (AgentHardenerRunJob, {"manifest_id": "m1", "validate_only": True}),
        (AgentHardenerRunJob, {"manifest_id": "m1", "config": "manifest.yaml"}),
        (AgentHardenerSynthBenignJob, {"manifest_id": ""}),
    ],
)
def test_worker_rejects_invalid_inputs_before_provisioning(
    job_class: type[AgentHardenerRunJob] | type[AgentHardenerSynthBenignJob],
    config: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provision_checks: list[object] = []
    monkeypatch.setattr(_common, "require_provisioned", provision_checks.append)
    with pytest.raises(ValidationError):
        job_class()._execute(config, ctx=make_job_context(tmp_path), openshell_home=tmp_path / "openshell")
    assert provision_checks == []
