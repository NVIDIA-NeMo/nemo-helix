# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Dockerfile an agent layer builds: the task's environment, then the agent installed into its final stage."""

import json

import pytest
from nemo_evals.agent_layers.dockerfile import (
    AGENT_MOUNT,
    INSTALL_MODULE,
    INSTALLER_MOUNT,
    final_stage_user,
    layer_dockerfile,
)


@pytest.mark.parametrize(
    ("dockerfile", "user"),
    [
        ("FROM ubuntu:24.04\nRUN true\n", None),
        ("FROM ubuntu:24.04\nUSER agent\n", "agent"),
        ("FROM ubuntu AS build\nUSER builder\nFROM ubuntu\nRUN true\n", None),
        ("FROM ubuntu\nUSER a\nRUN true\nUSER 1000:1000\n", "1000:1000"),
        ("FROM ubuntu\n# USER commented\nRUN echo \\\n  USER not-an-instruction\n", None),
        ("from ubuntu\nuser agent\n", "agent"),
    ],
    ids=["none", "set", "earlier-stage-only", "last-wins", "comment-and-continuation", "lowercase"],
)
def test_the_final_stages_user(dockerfile: str, user: str | None) -> None:
    assert final_stage_user(dockerfile) == user


def test_the_install_runs_as_root_in_the_final_stage_and_restores_its_user() -> None:
    original = "FROM ubuntu:24.04\nWORKDIR /app\nUSER agent\n"

    layered = layer_dockerfile(environment_dockerfile=original, base_image=None)

    assert layered.startswith(original.rstrip("\n"))
    tail = layered.splitlines()[-3:]
    assert tail[0] == "USER root"
    assert tail[1].startswith(
        f"RUN --mount=type=bind,source=nhx-agent-installer,target={INSTALLER_MOUNT} "
        f"--mount=type=bind,source=nhx-agent-layer,target={AGENT_MOUNT} "
    )
    command = json.loads(tail[1].split(" ", 3)[3])
    assert command == [f"{INSTALLER_MOUNT}/venv/bin/python", "-m", INSTALL_MODULE, f"{AGENT_MOUNT}/agent.json"]
    assert tail[2] == "USER agent"


def test_a_prebuilt_environment_starts_from_its_image() -> None:
    layered = layer_dockerfile(environment_dockerfile=None, base_image="docker.io/library/python:3.12")

    assert layered.splitlines()[0] == "FROM docker.io/library/python:3.12"
    assert layered.splitlines()[-1].startswith("RUN --mount=")  # no USER to restore


def test_exactly_one_base_is_given() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        layer_dockerfile(environment_dockerfile=None, base_image=None)
    with pytest.raises(ValueError, match="exactly one"):
        layer_dockerfile(environment_dockerfile="FROM a\n", base_image="b")
