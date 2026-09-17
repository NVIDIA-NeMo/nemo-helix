# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deriving a manifest from a project the platform did not render.

The contract under test is not "does it parse a Dockerfile" but "does it know what it does not
know": everything derivable is derived, and everything else lands in ``unresolved`` rather than
being silently defaulted into a run that fails twenty minutes later.
"""

from __future__ import annotations

from pathlib import Path

from nemo_agent_hardener_plugin.project_resolver import (
    derive_binaries,
    derive_port,
    derive_start_command,
    dockerfile_env,
    inspect_project,
)

FABRIC_DOCKERFILE = """\
FROM python:3.12-slim
RUN python -m venv /workspace/.venv \\
 && /workspace/.venv/bin/pip install --no-cache-dir "nemo-platform"
ENV AGENT_CONFIG_PATH=/workspace/agent.yaml
ENV PORT=8000
ENV PATH="/workspace/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/workspace/.venv
ENV INFERENCE_API_KEY=""
EXPOSE 8000
ENTRYPOINT ["sh", "-c", "exec python -m nemo_agents_plugin.fabric.server \
--agent-config \\"$AGENT_CONFIG_PATH\\" --host 0.0.0.0 --port \\"$PORT\\""]
"""


def _project(tmp_path: Path, dockerfile: str = FABRIC_DOCKERFILE, name: str = "Dockerfile") -> Path:
    (tmp_path / name).write_text(dockerfile, encoding="utf-8")
    return tmp_path


def test_env_parses_both_dockerfile_forms():
    env = dockerfile_env('ENV A=1 B="two"\nENV LEGACY three\nENV C=4\n')
    assert env == {"A": "1", "B": "two", "LEGACY": "three", "C": "4"}


def test_start_command_is_absolute_because_openshell_replaces_path():
    """A bare interpreter resolves through the image's PATH, which the sandbox does not keep.

    Left bare, the victim never starts and the run reports an uninstrumented victim — an error that
    says nothing about the command being wrong.
    """
    env = dockerfile_env(FABRIC_DOCKERFILE)
    command = derive_start_command(FABRIC_DOCKERFILE, env)

    assert command.startswith("/workspace/.venv/bin/python ")
    # $VAR references resolved from the image's own ENV, since the sandbox does not propagate them.
    assert "$" not in command
    assert "--agent-config /workspace/agent.yaml" in command
    assert "--port 8000" in command


def test_start_command_is_unknown_for_a_shell_form():
    """A shell-form ENTRYPOINT is not machine-readable, and guessing would be a guess."""
    assert derive_start_command("FROM x\nENTRYPOINT python -m app\n", {}) == ""


def test_start_command_is_unknown_when_a_variable_cannot_be_resolved():
    text = 'FROM x\nENTRYPOINT ["sh","-c","exec python -m app --cfg $MISSING"]\n'
    assert derive_start_command(text, {"VIRTUAL_ENV": "/v"}) == ""


def test_binaries_include_the_system_interpreter_the_venv_resolves_to():
    """OpenShell matches on the *resolved* executable, and venv/bin/python is a symlink.

    A block naming only the venv glob matches no process: it grants nothing while looking like it
    grants something, which is the worst available failure shape.
    """
    binaries = derive_binaries(FABRIC_DOCKERFILE, dockerfile_env(FABRIC_DOCKERFILE))
    assert "/workspace/.venv/bin/**" in binaries
    assert "/usr/local/bin/python*" in binaries


def test_port_prefers_env_over_expose():
    assert derive_port("FROM x\nENV PORT=9000\nEXPOSE 8000\n", {"PORT": "9000"}) == 9000
    assert derive_port("FROM x\nEXPOSE 8080/tcp\n", {}) == 8080
    assert derive_port("FROM x\n", {}) == 8000


def test_inspect_derives_everything_a_fabric_image_states(tmp_path: Path):
    derived = inspect_project(_project(tmp_path))

    assert derived["dockerfile"] == "Dockerfile"
    assert derived["port"] == 8000
    assert derived["start_command"].startswith("/workspace/.venv/bin/python ")
    # Credentials are named, never carried: the value is empty in the image and must not be copied on.
    assert derived["secrets"] == ["INFERENCE_API_KEY"]
    assert "INFERENCE_API_KEY" not in derived["env"]
    assert derived["env"]["AGENT_CONFIG_PATH"] == "/workspace/agent.yaml"


def test_unresolved_names_only_what_the_project_cannot_state(tmp_path: Path):
    """Harness and Relay attachment are facts about how the agent is wired, not about its files."""
    derived = inspect_project(_project(tmp_path))
    assert derived["unresolved"] == ["harness", "relay_integration_confirmed"]


def test_a_shell_form_entrypoint_adds_start_command_to_unresolved(tmp_path: Path):
    derived = inspect_project(_project(tmp_path, "FROM x\nENV VIRTUAL_ENV=/v\nENTRYPOINT python -m app\n"))
    assert "start_command" in derived["unresolved"]
    assert any("shell form" in warning for warning in derived["warnings"])


def test_ambiguous_dockerfile_asks_instead_of_picking(tmp_path: Path):
    """Choosing for the user would war-game whichever image sorted first."""
    (tmp_path / "Dockerfile").write_text("FROM a\n", encoding="utf-8")
    (tmp_path / "Dockerfile.dev").write_text("FROM b\n", encoding="utf-8")

    derived = inspect_project(tmp_path)
    assert derived["unresolved"] == ["dockerfile"]
    assert sorted(derived["dockerfiles"]) == ["Dockerfile", "Dockerfile.dev"]


def test_a_named_dockerfile_resolves_the_ambiguity(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text("FROM a\n", encoding="utf-8")
    (tmp_path / "Dockerfile.dev").write_text(FABRIC_DOCKERFILE, encoding="utf-8")

    derived = inspect_project(tmp_path, dockerfile="Dockerfile.dev")
    assert derived["dockerfile"] == "Dockerfile.dev"
    assert "dockerfile" not in derived["unresolved"]


def test_an_absolute_dockerfile_path_is_refused(tmp_path: Path):
    """`project_root / dockerfile` silently discards project_root for an absolute right side."""
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("outside the bundle", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text(FABRIC_DOCKERFILE, encoding="utf-8")

    derived = inspect_project(tmp_path, dockerfile=str(secret))
    assert derived["unresolved"] == ["dockerfile"]
    assert any("not a file in the uploaded bundle" in warning for warning in derived["warnings"])


def test_a_dockerfile_path_that_escapes_the_bundle_is_refused(tmp_path: Path):
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("outside the bundle", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text(FABRIC_DOCKERFILE, encoding="utf-8")

    derived = inspect_project(tmp_path, dockerfile=f"../{secret.name}")
    assert derived["unresolved"] == ["dockerfile"]
    assert any("not a file in the uploaded bundle" in warning for warning in derived["warnings"])


def test_a_named_dockerfile_resolves_through_a_symlinked_project_root(tmp_path: Path):
    """The containment check must not leave `chosen` resolved while project_root is not.

    macOS puts temp dirs under /var, which is a symlink to /private/var — and the API inspects an
    uploaded bundle from exactly such a temp dir. Comparing a resolved path against an unresolved
    root made `relative_to` raise, surfacing as a 400 "could not read the project bundle".
    """
    real = tmp_path / "real_bundle"
    real.mkdir()
    (real / "Dockerfile").write_text(FABRIC_DOCKERFILE, encoding="utf-8")
    linked = tmp_path / "linked_bundle"
    linked.symlink_to(real, target_is_directory=True)

    derived = inspect_project(linked, dockerfile="Dockerfile")

    assert derived["dockerfile"] == "Dockerfile"
    assert "dockerfile" not in derived["unresolved"]


def test_vendored_dockerfiles_do_not_create_ambiguity(tmp_path: Path):
    """A Dockerfile under .venv or node_modules is not the agent's own."""
    (tmp_path / "Dockerfile").write_text(FABRIC_DOCKERFILE, encoding="utf-8")
    vendored = tmp_path / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "Dockerfile").write_text("FROM other\n", encoding="utf-8")

    assert inspect_project(tmp_path)["dockerfile"] == "Dockerfile"


def test_a_project_with_no_dockerfile_is_refused(tmp_path: Path):
    """Agent Hardener hardens the image you ship; it does not build one."""
    derived = inspect_project(tmp_path)
    assert derived["unresolved"] == ["dockerfile"]
    assert any("builds none" in warning for warning in derived["warnings"])


def test_missing_secrets_and_egress_are_warned_not_invented(tmp_path: Path):
    """Both fail mid-run, and the sandbox is default-deny, so silence here is expensive."""
    derived = inspect_project(_project(tmp_path, 'FROM x\nENV VIRTUAL_ENV=/v\nENTRYPOINT ["/v/bin/python"]\n'))
    assert derived["secrets"] == []
    assert derived["egress"] == []
    assert any("secrets" in warning for warning in derived["warnings"])
    assert any("default-deny" in warning for warning in derived["warnings"])


def test_egress_is_taken_from_hosts_the_project_names(tmp_path: Path):
    text = "FROM x\nENV BASE_URL=https://inference-api.nvidia.com/v1\nRUN curl https://pypi.org/simple\n"
    derived = inspect_project(_project(tmp_path, text))
    # Exact list, not membership: `"host" in ...` reads to CodeQL as URL-substring
    # sanitization, and asserting the whole sorted list is a stronger check anyway.
    assert derived["egress"] == ["inference-api.nvidia.com", "pypi.org"]


def test_secrets_come_from_a_committed_dotenv_too(tmp_path: Path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-x\n# comment\nPLAIN=1\n", encoding="utf-8")
    derived = inspect_project(_project(tmp_path))
    assert "OPENAI_API_KEY" in derived["secrets"]
    assert "PLAIN" in derived["secrets"]  # a name in the agent's dotenv is a name it needs at run time
