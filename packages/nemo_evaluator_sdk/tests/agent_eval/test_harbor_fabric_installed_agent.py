# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The installed Fabric agent's contract: install into any image, then behave like ``FabricAgent``."""

import ast
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

pytest.importorskip("harbor", reason="FabricInstalledAgent builds on Harbor's BaseInstalledAgent")

import harbor
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent import NemoFabricAgent
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_installed_agent import (
    DEFAULT_FABRIC_MAX_TURNS,
    DEFAULT_UV_VERSION,
    FabricInstalledAgent,
)

_DEEPAGENTS = "nvidia.fabric.langchain.deepagents"
_PACKAGE = "nemo-fabric[deepagents,relay]==0.3.0"


class _ExecResult:
    def __init__(self, return_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class _RecordingEnvironment:
    """The slice of ``BaseEnvironment`` ``install()`` touches, recording what it was asked to do."""

    default_user = "agent"
    session_id = "hello-world__aaa"

    def __init__(self) -> None:
        self.commands: list[tuple[str, str | int | None]] = []
        self.uploaded_dirs: list[tuple[Path, str]] = []

    async def exec(
        self,
        command: str,
        user: str | int | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> _ExecResult:
        del env, cwd, timeout_sec
        self.commands.append((command, user))
        return _ExecResult()

    async def upload_dir(self, source_path: Path, target_path: str) -> None:
        self.uploaded_dirs.append((source_path, target_path))


def _agent(tmp_path: Path, **kwargs: Any) -> FabricInstalledAgent:
    return FabricInstalledAgent(
        logs_dir=tmp_path,
        fabric_adapter_id=_DEEPAGENTS,
        fabric_package=_PACKAGE,
        fabric_workspace="/app",
        **kwargs,
    )


def _ran(environment: _RecordingEnvironment, needle: str) -> str:
    matches = [command for command, _ in environment.commands if needle in command]
    assert matches, f"no command contained {needle!r}; ran: {[c for c, _ in environment.commands]}"
    return matches[0]


def test_it_is_an_installed_agent_wrapping_a_fabric_agent(tmp_path: Path) -> None:
    """Harbor's install scaffolding on the outside, Fabric's run protocol on the inside."""
    agent = _agent(tmp_path, model_name="nvidia/nemotron-3.5-lightning-30b-a3b")

    assert isinstance(agent, BaseInstalledAgent)
    assert isinstance(agent.fabric, NemoFabricAgent)
    # Composition, so FabricAgent.setup() -- the `python3 -m venv` step that cannot run on a bare
    # image -- is never reached, and the model config still comes from the wrapped agent.
    assert type(agent).setup is BaseInstalledAgent.setup
    assert agent.SUPPORTS_ATIF is True
    assert agent.fabric._build_spec("write hello.txt").config.models["default"].api_key_env == "NVIDIA_API_KEY"


async def test_state_harbor_assigns_after_construction_reaches_the_wrapped_agent(tmp_path: Path) -> None:
    """The tripwire for composition's one real risk.

    Harbor sets these on the object it constructed, after construction: `session_id`/`context_id`
    (trial.py, regrade.py) and `skills` (trial.py, job.py, job_plan.py). `skills` is not declared on
    `BaseAgent` at all -- Harbor attaches it dynamically -- so if Harbor grows a fourth such field
    nothing but this test will notice. Extend it rather than debugging a silently empty run context.
    """
    agent = _agent(tmp_path)
    context_id = uuid4()
    seen: dict[str, object] = {}

    async def _capture(instruction: str, environment: object, context: object) -> None:
        del instruction, environment, context
        seen.update(
            session_id=agent.fabric.session_id,
            context_id=agent.fabric.context_id,
            skills=getattr(agent.fabric, "skills", None),
        )

    agent.fabric.run = _capture
    agent.session_id = "hello-world__aaa__agent"
    agent.context_id = context_id
    agent.skills = ["/skills/researching"]

    await agent.run("write hello.txt", cast(BaseEnvironment, _RecordingEnvironment()), cast(Any, None))

    assert seen == {
        "session_id": "hello-world__aaa__agent",
        "context_id": context_id,
        "skills": ["/skills/researching"],
    }


def test_the_session_ids_reach_the_fabric_run_request(tmp_path: Path) -> None:
    """What the copying is actually for: correlating a Fabric run with its Harbor trial."""
    agent = _agent(tmp_path)
    context_id = uuid4()
    agent.session_id = "hello-world__aaa__agent"
    agent.context_id = context_id

    agent._copy_harbor_assigned_state()
    request_context = agent.fabric._build_spec("write hello.txt").request.context

    assert request_context["harbor_session_id"] == "hello-world__aaa__agent"
    assert request_context["harbor_context_id"] == str(context_id)


async def test_install_provisions_curl_uv_and_the_fabric_venv_in_order(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    environment = _RecordingEnvironment()

    await agent.install(cast(BaseEnvironment, environment))

    _ran(environment, "apt-get install -y curl ca-certificates")
    prepare = _ran(environment, "mkdir -p")
    assert "/logs/agent" in prepare and "/tmp/nemo-fabric-venv" in prepare
    assert "chown -R agent:agent" in prepare
    install = _ran(environment, "astral.sh/uv")
    assert "uv python install 3.12" in install
    assert "uv venv /tmp/nemo-fabric-venv --python 3.12 --clear" in install
    # The venv's python, never the image's: that is what makes an arbitrary task image workable.
    assert f"uv pip install --python /tmp/nemo-fabric-venv/bin/python '{_PACKAGE}'" in install
    assert environment.commands.index((prepare, "root")) < environment.commands.index((install, None))


async def test_the_uv_installer_is_pinned_to_a_release(tmp_path: Path) -> None:
    """Unpinned, the same eval run a week later provisions a different toolchain.

    Every upstream Harbor installed agent fetches the unversioned installer; pinning is what makes
    the provisioned toolchain a property of the config rather than of the day it ran.
    """
    environment = _RecordingEnvironment()

    await _agent(tmp_path).install(cast(BaseEnvironment, environment))

    assert f"https://astral.sh/uv/{DEFAULT_UV_VERSION}/install.sh" in _ran(environment, "astral.sh/uv")

    pinned = _RecordingEnvironment()
    await _agent(tmp_path, fabric_uv_version="0.10.10").install(cast(BaseEnvironment, pinned))

    assert "https://astral.sh/uv/0.10.10/install.sh" in _ran(pinned, "astral.sh/uv")


def test_an_unquoted_yaml_uv_version_is_rejected(tmp_path: Path) -> None:
    """`uv_version: 0.12` parses as the float 0.12, which is not the release `0.12.17`."""
    with pytest.raises(ValueError, match="fabric_uv_version must be a string"):
        _agent(tmp_path, fabric_uv_version=0.12)


async def test_the_package_manager_step_retries_with_backoff(tmp_path: Path) -> None:
    """A blipping archive mirror should not cost the trial.

    Harbor's own retry restarts the whole trial, so a one-off `apt-get` failure otherwise throws
    away the run. The retry is in-shell -- Harbor gives the host a single exec -- so the command
    string is the only observable surface.
    """
    environment = _RecordingEnvironment()

    await _agent(tmp_path).install(cast(BaseEnvironment, environment))

    provision = _ran(environment, "ca-certificates")
    assert 'if eval "$install"; then break; fi' in provision, "the chosen manager is not retried"
    assert '[ "$attempt" -ge 3 ]' in provision, "retries are not bounded"
    assert "sleep $((attempt * 5))" in provision, "retries do not back off"
    # Selection stays outside the loop: an image with no package manager will never grow one.
    assert provision.index("no supported package manager") < provision.index("attempt=1")


async def test_the_uv_installer_fetch_is_retried_by_curl(tmp_path: Path) -> None:
    """`--retry-all-errors` is probed, not assumed: it landed in curl 7.71 and bullseye ships older."""
    environment = _RecordingEnvironment()

    await _agent(tmp_path).install(cast(BaseEnvironment, environment))

    install = _ran(environment, "astral.sh/uv")
    assert "curl -LsSf --retry 5 --retry-delay 2 $retry_all" in install
    assert "if curl --help all 2>/dev/null | grep -q -- --retry-all-errors; then" in install


async def test_install_runs_as_root_for_packages_and_as_the_agent_for_uv(tmp_path: Path) -> None:
    """uv writes into ``$HOME``; running it as root would install into the wrong home."""
    agent = _agent(tmp_path)
    environment = _RecordingEnvironment()

    await agent.install(cast(BaseEnvironment, environment))

    users = {command: user for command, user in environment.commands}
    assert users[_ran(environment, "mkdir -p")] == "root"
    assert users[_ran(environment, "astral.sh/uv")] is None


async def test_a_config_bundle_is_uploaded_and_its_target_created(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "adapters").mkdir(parents=True)
    agent = _agent(tmp_path, fabric_config_bundle=bundle, fabric_config_target="/tmp/fabric-config")
    environment = _RecordingEnvironment()

    await agent.install(cast(BaseEnvironment, environment))

    assert environment.uploaded_dirs == [(bundle, "/tmp/fabric-config")]
    assert "/tmp/fabric-config" in _ran(environment, "mkdir -p")


async def test_every_supported_package_manager_installs_ca_certificates(tmp_path: Path) -> None:
    """CA certificates matter as much as curl.

    ``ubuntu:24.04`` ships neither, and without the certificates the uv installer's HTTPS fetch
    fails with a bare curl exit code that reads like a network outage.
    """
    agent = _agent(tmp_path)
    environment = _RecordingEnvironment()

    await agent.install(cast(BaseEnvironment, environment))

    provision = _ran(environment, "ca-certificates")
    for manager in ("apt-get", "apk", "dnf", "yum"):
        assert f"command -v {manager}" in provision, f"{manager} is not probed"
    assert provision.count("ca-certificates") == 4, "every manager branch must install them"


def test_fabric_package_is_required(tmp_path: Path) -> None:
    """Without it the wrapped agent's ``_runner_python`` falls back to the image's ``python3``."""
    with pytest.raises(ValueError, match="fabric_package is required"):
        FabricInstalledAgent(logs_dir=tmp_path, fabric_adapter_id=_DEEPAGENTS)


def test_an_unquoted_yaml_python_version_is_rejected(tmp_path: Path) -> None:
    """YAML reads `python_version: 3.10` as the float 3.1, which would install Python 3.1."""
    with pytest.raises(ValueError, match="quote it as '3.1'"):
        _agent(tmp_path, fabric_python_version=3.1)


def test_the_harness_gets_a_turn_budget_by_default(tmp_path: Path) -> None:
    """Fabric leaves the harness unbounded, and an unbounded harness loses the whole trial.

    Harbor kills an over-running agent phase, and a killed phase writes no `RunResult` at all -- no
    trajectory, no error, just `AgentTimeoutError`. Six of ten `terminal-bench-sample` trials died
    that way before this default existed.
    """
    assert _agent(tmp_path).fabric.fabric_max_turns == DEFAULT_FABRIC_MAX_TURNS


def test_an_explicit_turn_budget_wins_including_unbounded(tmp_path: Path) -> None:
    """`None` is a real choice (Fabric's own behaviour), not an absent argument."""
    assert _agent(tmp_path, fabric_max_turns=5).fabric.fabric_max_turns == 5
    assert _agent(tmp_path, fabric_max_turns=None).fabric.fabric_max_turns is None


def test_the_version_probe_reads_the_venv_not_the_host(tmp_path: Path) -> None:
    command = _agent(tmp_path).get_version_command()

    assert command is not None
    assert command.startswith("/tmp/nemo-fabric-venv/bin/python -c")
    assert "nemo-fabric-runtime" in command


# --- Drift guard: what Harbor assigns onto an agent it constructed --------------------------------

#: Attributes Harbor assigns onto an `agent`/`user_agent` that this wrapper deliberately ignores.
#: Anything not here and not in `_HARBOR_ASSIGNED_ATTRIBUTES` fails the scan below, so a new field
#: has to be classified rather than silently dropped on the floor.
_NOT_OUR_AGENT_STATE = {
    "secrets": "hosted-job CLI config objects (`harbor.cli.hosted_jobs`), not BaseAgent instances",
    "n_concurrent": "`for agent in config.agents` in the jobs CLI -- AgentConfig models, not instances",
    "resume_trajectory": "`for agent in config.agents` in the jobs CLI -- AgentConfig models, not instances",
    "_n_episodes": "computer_1's provider writing to its own agent class",
    "_early_termination_reason": "computer_1's provider writing to its own agent class",
}


def _attributes_harbor_assigns_on_agents() -> dict[str, str]:
    """Parse Harbor and return ``{attribute: "file:line"}`` for every ``<...>agent.attr = ...``.

    An AST walk rather than a grep: `agent.name=` also appears inside an f-string, and a regex
    cannot tell that from an assignment.
    """
    harbor_root = Path(harbor.__file__).parent
    found: dict[str, str] = {}
    for path in sorted(harbor_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:  # pragma: no cover - vendored templates need not be importable
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                owner = target.value
                # A constructed agent is held as a local `agent` or as `self.agent`. Deliberately
                # NOT `config.agent`, which is the declarative `AgentConfig` on a TrialConfig and
                # carries a whole different set of fields (`import_path`, `override_timeout_sec`...).
                is_instance = isinstance(owner, ast.Name) and owner.id in {"agent", "user_agent"}
                is_self_held = (
                    isinstance(owner, ast.Attribute)
                    and isinstance(owner.value, ast.Name)
                    and owner.value.id == "self"
                    and owner.attr in {"agent", "user_agent"}
                )
                if is_instance or is_self_held:
                    found.setdefault(target.attr, f"{path.relative_to(harbor_root)}:{target.lineno}")
    return found


def test_every_attribute_harbor_assigns_on_an_agent_is_classified() -> None:
    """Composition's structural risk, made loud.

    Harbor sets state on the object it constructed; a wrapper only sees it if this class copies it.
    `skills` is not declared on `BaseAgent` at all, so there is no base class to diff against -- this
    scan is the substitute. When Harbor grows a fourth such field, classify it: forward it by adding
    it to `_HARBOR_ASSIGNED_ATTRIBUTES`, or record why it is irrelevant in `_NOT_OUR_AGENT_STATE`.
    """
    assigned = _attributes_harbor_assigns_on_agents()
    # The scan must keep finding the fields we already know about, or it has silently stopped working.
    assert set(FabricInstalledAgent._HARBOR_ASSIGNED_ATTRIBUTES) <= set(assigned), (
        f"scan found {sorted(assigned)}, which no longer covers the forwarded attributes; "
        "the parse or Harbor's call sites changed"
    )

    unclassified = set(assigned) - set(FabricInstalledAgent._HARBOR_ASSIGNED_ATTRIBUTES) - set(_NOT_OUR_AGENT_STATE)

    assert not unclassified, (
        f"Harbor now assigns {sorted((a, assigned[a]) for a in unclassified)} onto an agent. "
        "Forward it in _HARBOR_ASSIGNED_ATTRIBUTES, or explain the exclusion in _NOT_OUR_AGENT_STATE."
    )
