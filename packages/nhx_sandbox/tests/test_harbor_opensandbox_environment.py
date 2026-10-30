# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NemoOpenSandboxEnvironment against a mock OpenSandbox SDK.

Written to the Harbor environment API that Harbor 0.13.2 and 0.20.0 share, so the same tests run
against either runner's Harbor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("harbor.environments.base")
pytest.importorskip("nhx_sandbox")

from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from nhx_sandbox import harbor_opensandbox_environment as environment_module
from nhx_sandbox.core.termination import TERMINATION_GLOB
from nhx_sandbox.harbor_opensandbox_cleanup import APPLIED_EGRESS_GLOB, applied_egress_filename
from nhx_sandbox.harbor_opensandbox_driver import OpenSandboxDriver, SandboxExpiredError
from nhx_sandbox.harbor_opensandbox_environment import (
    CREATE_ATTEMPT_METADATA_KEY,
    MANAGED_BY_METADATA_KEY,
    MANAGED_BY_METADATA_VALUE,
    NemoOpenSandboxEnvironment,
    sandbox_role,
)
from opensandbox_mocks import Sandbox, SandboxApiException, Server, execution, mock_sdk

# The runner's trusted egress allowlist: an IP, a wildcard hostname and an IPv6 CIDR, one of each form.
TRUSTED = ["10.96.0.20", "*.nvidia.com", "2001:db8::/48"]
# Trial name, which Harbor also uses for the trial directory and the sandbox session IDs.
TRIAL = "hello-world__abc"


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Server:
    """A fresh mock OpenSandbox server, which also answers the environment's own requests to the server API."""
    server = Server()
    real_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        transport = httpx.MockTransport(
            lambda request: (
                server.endpoint_route(request)
                or server.sandbox_route(request)
                or server.events_route(request)
                or httpx.Response(404)
            )
        )
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(environment_module.httpx, "AsyncClient", client)
    monkeypatch.setattr(environment_module, "SERVER_RETRY_BASE_SEC", 0)
    return server


def _environment(
    tmp_path: Path,
    server: Server,
    mode: NetworkMode,
    allowed_hosts: list[str] | None = None,
    *,
    task: dict[str, Any] | None = None,
    trial_name: str = TRIAL,
    session_id: str | None = None,
    **kwargs: Any,
) -> NemoOpenSandboxEnvironment:
    """An environment for a hello-world task on the mock server, with the trusted allowlist above.

    ``task`` adds fields to the task's ``[environment]``; ``kwargs`` go to the environment. The
    session ID defaults to the agent's under Harbor 0.13.2 naming: the bare trial name.
    """
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(exist_ok=True)
    environment = NemoOpenSandboxEnvironment(
        environment_dir=environment_dir,
        environment_name="hello-world",
        session_id=session_id or trial_name,
        trial_paths=TrialPaths(trial_dir=tmp_path / trial_name),
        task_env_config=EnvironmentConfig(docker_image="registry.example/hello:1", **(task or {})),
        network_policy=NetworkPolicy(network_mode=mode, allowed_hosts=allowed_hosts or []),
        domain="opensandbox.example",
        api_key="",
        metadata={"nemo-scaled-evals-run": "run-1"},
        trusted_allowed_hosts=TRUSTED,
        resolver_addresses=["10.96.0.10"],
        **kwargs,
    )
    environment._driver = OpenSandboxDriver(
        environment._connection, sdk=mock_sdk(server), logger=environment.logger, retry_wait=lambda attempt: 0
    )
    return environment


def _policy_sent(server: Server) -> dict[str, Any]:
    """The egress policy payload the last create sent."""
    return server.creates[-1]["network_policy"].payload


def _sandbox(environment: NemoOpenSandboxEnvironment) -> Sandbox:
    """The environment's live sandbox, which in these tests is always the mock's."""
    sandbox = environment._sandbox
    assert isinstance(sandbox, Sandbox)
    return sandbox


def _imports_with_blocked_modules(module: str, blocked: list[str]) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a fresh interpreter where importing any of ``blocked`` fails."""
    # A None entry in sys.modules makes Python raise ImportError for that module and its submodules.
    script = "import sys\nfor name in sys.argv[2:]: sys.modules[name] = None\nimport importlib; importlib.import_module(sys.argv[1])"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    return subprocess.run(
        [sys.executable, "-c", script, module, *blocked], env=env, capture_output=True, text=True, check=False
    )


def test_environment_does_not_need_harbors_own_opensandbox_environment() -> None:
    """The point of this class: it loads on a Harbor release that has no OpenSandbox environment."""
    result = _imports_with_blocked_modules(
        "nhx_sandbox.harbor_opensandbox_environment", ["harbor.environments.opensandbox"]
    )

    assert result.returncode == 0, result.stderr
    assert NemoOpenSandboxEnvironment.__mro__[1] is BaseEnvironment


def test_driver_does_not_need_harbor() -> None:
    """The driver is Harbor-free, so the shared package's runner-side tools can import it without Harbor."""
    result = _imports_with_blocked_modules("nhx_sandbox.harbor_opensandbox_driver", ["harbor"])

    assert result.returncode == 0, result.stderr


async def test_no_network_is_deny_all(tmp_path: Path, server: Server) -> None:
    """A ``no-network`` task gets a deny-all policy with no exceptions."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)

    await environment._create_sandbox()

    assert _policy_sent(server) == {"defaultAction": "deny", "egress": []}


async def test_public_gets_the_trusted_allowlist_not_unrestricted_access(tmp_path: Path, server: Server) -> None:
    """A ``public`` task gets the trusted allowlist plus the resolver, never unrestricted egress."""
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)

    await environment._create_sandbox()

    policy = _policy_sent(server)
    allowed = {rule["target"] for rule in policy["egress"] if rule["action"] == "allow"}
    denied = {rule["target"] for rule in policy["egress"] if rule["action"] == "deny"}
    assert policy["defaultAction"] == "deny"
    assert allowed == {*TRUSTED, "10.96.0.10"}
    assert "192.168.0.0/16" in denied
    assert "10.0.0.0/8" not in denied


async def test_records_the_verified_policy_for_the_supervisor(tmp_path: Path, server: Server) -> None:
    """The verified policy, its hash, the session and the role are recorded in the trial directory."""
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)

    sandbox = await environment._create_sandbox()

    record = json.loads((tmp_path / TRIAL / applied_egress_filename(sandbox.id)).read_text())
    canonical = json.dumps(record["policy"], sort_keys=True, separators=(",", ":"))
    assert record["sandbox_id"] == sandbox.id
    assert record["session_id"] == TRIAL
    assert record["role"] == "agent"
    assert record["network_mode"] == "public"
    assert record["policy"] == _policy_sent(server)
    assert record["policy_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("agent_session", [TRIAL, f"{TRIAL}__env"], ids=["harbor-0.13", "harbor-0.20"])
async def test_agent_and_separate_verifier_each_keep_their_record(
    tmp_path: Path, server: Server, agent_session: str
) -> None:
    """An agent and a separate verifier in one trial each keep their own egress record, under either release's naming."""
    agent = _environment(tmp_path, server, NetworkMode.PUBLIC, session_id=agent_session)
    verifier = _environment(tmp_path, server, NetworkMode.NO_NETWORK, session_id=f"{TRIAL}__verifier__task")

    agent_sandbox = await agent._create_sandbox()
    verifier_sandbox = await verifier._create_sandbox()

    records = {
        record["sandbox_id"]: record
        for record in (json.loads(path.read_text()) for path in (tmp_path / TRIAL).glob(APPLIED_EGRESS_GLOB))
    }
    assert set(records) == {agent_sandbox.id, verifier_sandbox.id}
    assert records[agent_sandbox.id]["role"] == "agent"
    assert records[verifier_sandbox.id]["role"] == "verifier"
    assert records[verifier_sandbox.id]["network_mode"] == "no-network"


@pytest.mark.parametrize(
    ("trial_name", "session_id", "role"),
    [
        # Harbor 0.13.2 names the agent's session after the trial; 0.20 adds __env.
        ("trial-1", "trial-1", "agent"),
        ("trial-1", "trial-1__env", "agent"),
        ("trial-1", "trial-1__verifier__task", "verifier"),
        ("trial-1", "trial-1__verifier__env", "verifier"),
        ("trial-1-with-a-long-name", "trial-1-with-a-long-name__verifier__step-o__1a2b3c4d", "verifier"),
        # A trial name may itself contain __verifier__; only an exact match is the agent.
        ("a__verifier__b__xyz", "a__verifier__b__xyz", "agent"),
        ("a__verifier__b__xyz", "a__verifier__b__xyz__env", "agent"),
        ("a__verifier__b__xyz", "a__verifier__b__xyz__verifier__env", "verifier"),
    ],
)
def test_sandbox_role_from_harbor_session_id(trial_name: str, session_id: str, role: str) -> None:
    """Only the agent's exact session ID counts as the agent; every other session is a verifier."""
    assert sandbox_role(session_id, trial_name) == role


async def test_agent_is_recorded_as_agent_when_the_trial_name_contains_verifier(tmp_path: Path, server: Server) -> None:
    """A trial name containing ``__verifier__`` doesn't make the agent look like a verifier."""
    trial = "task__verifier__x__Ab12Cd3"
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC, trial_name=trial)

    sandbox = await environment._create_sandbox()

    record = json.loads((tmp_path / trial / applied_egress_filename(sandbox.id)).read_text())
    assert record["role"] == "agent"


def test_harbor_session_naming_still_matches_sandbox_role() -> None:
    """``sandbox_role`` relies on the installed Harbor's session naming; a release that changes it must fail here."""
    import inspect

    from harbor.trial.trial import Trial

    source = inspect.getsource(Trial)
    assert "}__verifier__{" in inspect.getsource(Trial._separate_verifier_session_id)
    # The agent's session: the bare trial name (0.13.2) or the trial name plus __env (0.20).
    assert "session_id=self.config.trial_name," in source or 'session_id=f"{self.config.trial_name}__env"' in source


async def test_allowlist_task_narrows_the_trusted_allowlist(tmp_path: Path, server: Server) -> None:
    """An ``allowlist`` task gets only the hosts it asked for, plus the resolver."""
    environment = _environment(tmp_path, server, NetworkMode.ALLOWLIST, ["*.NVIDIA.com"])

    await environment._create_sandbox()

    allowed = {rule["target"] for rule in _policy_sent(server)["egress"] if rule["action"] == "allow"}
    assert allowed == {"*.NVIDIA.com", "10.96.0.10"}


def test_allowlist_task_cannot_extend_the_trusted_allowlist(tmp_path: Path, server: Server) -> None:
    """A task asking for a host outside the trusted allowlist is rejected before any sandbox exists."""
    with pytest.raises(ValueError, match=r"\['pypi.org'\].*does not include"):
        _environment(tmp_path, server, NetworkMode.ALLOWLIST, ["*.nvidia.com", "pypi.org"])

    assert server.creates == []


def test_advertises_static_allowlist_support(tmp_path: Path, server: Server) -> None:
    """Harbor sees allowlist support for every entry form, but no dynamic policy changes."""
    capabilities = _environment(tmp_path, server, NetworkMode.PUBLIC).capabilities

    assert capabilities.network_allowlist
    assert capabilities.disable_internet
    assert not capabilities.dynamic_network_policy
    # Per-entry-type fields exist only on Harbor 0.20; where they exist, every type is supported.
    for name in type(capabilities).model_fields:
        if name.startswith("network_allowlist_"):
            assert getattr(capabilities, name), name


def test_reports_the_opensandbox_type(tmp_path: Path, server: Server) -> None:
    """The environment reports Harbor's ``opensandbox`` type."""
    assert _environment(tmp_path, server, NetworkMode.PUBLIC).type() == "opensandbox"


async def test_labels_every_create(tmp_path: Path, server: Server) -> None:
    """Every create carries the caller's labels, the session, the managed-by label and an attempt ID."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)

    await environment._create_sandbox()

    metadata = server.creates[-1]["metadata"]
    assert metadata["nemo-scaled-evals-run"] == "run-1"
    assert metadata["session_id"] == TRIAL
    assert metadata[MANAGED_BY_METADATA_KEY] == MANAGED_BY_METADATA_VALUE
    assert len(metadata[CREATE_ATTEMPT_METADATA_KEY]) == 32
    assert environment._metadata == {"nemo-scaled-evals-run": "run-1"}


async def test_sends_task_resources_env_and_gpu_types(tmp_path: Path, server: Server) -> None:
    """Task CPU, memory, GPU, GPU types, env and the sandbox lifetime reach the create call."""
    environment = _environment(
        tmp_path,
        server,
        NetworkMode.NO_NETWORK,
        task={"cpus": 2, "memory_mb": 4096, "gpus": 1, "gpu_types": ["h100"], "env": {"TASK_VAR": "x"}},
        sandbox_timeout_sec=600,
    )

    await environment._create_sandbox()

    sent = server.creates[-1]
    assert sent["image"] == "registry.example/hello:1"
    assert sent["resource"] == {"cpu": "2", "memory": "4096Mi", "gpu": "1"}
    assert sent["extensions"] == {"harbor.gpu_types": "h100"}
    assert sent["env"]["TASK_VAR"] == "x"
    assert sent["timeout"].total_seconds() == 600


async def test_a_sandbox_created_later_still_ends_at_the_expiry(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each create cuts the lifetime to the time left, so a later sandbox (a retry, a separate verifier) built
    from the same settings ends when the first one does, not ``sandbox_timeout_sec`` after its own create."""
    settings: dict[str, Any] = {"sandbox_timeout_sec": 3600, "sandbox_expires_at": "2026-10-23T13:00:00+00:00"}
    now = [datetime(2026, 10, 23, 12, 30, tzinfo=UTC).timestamp()]
    monkeypatch.setattr(time, "time", lambda: now[0])

    await _environment(tmp_path, server, NetworkMode.NO_NETWORK, **settings)._create_sandbox()
    now[0] += 20 * 60
    await _environment(tmp_path, server, NetworkMode.NO_NETWORK, trial_name="later", **settings)._create_sandbox()
    now[0] += 20 * 60  # past the expiry: refused before any create is sent
    with pytest.raises(SandboxExpiredError):
        await _environment(tmp_path, server, NetworkMode.NO_NETWORK, trial_name="late", **settings)._create_sandbox()

    assert [create["timeout"].total_seconds() for create in server.creates[-2:]] == [1800, 600]
    assert len(server.creates) == 2


async def test_without_an_expiry_every_sandbox_gets_the_full_lifetime(tmp_path: Path, server: Server) -> None:
    await _environment(tmp_path, server, NetworkMode.NO_NETWORK, sandbox_timeout_sec=3600)._create_sandbox()
    assert server.creates[-1]["timeout"].total_seconds() == 3600


async def test_kills_the_sandbox_when_the_applied_default_is_wrong(tmp_path: Path, server: Server) -> None:
    """A sandbox whose applied default action differs from the requested one is killed."""
    server.applied_default = "allow"
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)

    with pytest.raises(RuntimeError, match="applied egress default_action='allow'"):
        await environment._create_sandbox()

    assert server.live() == set()


async def test_kills_the_sandbox_when_the_policy_cannot_be_read_back(tmp_path: Path, server: Server) -> None:
    """A sandbox that can't report its applied policy is killed rather than trusted."""
    server.readback = False
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)

    with pytest.raises(RuntimeError, match="cannot report its applied egress policy"):
        await environment._create_sandbox()

    assert server.live() == set()


async def test_strict_verification_rejects_a_dropped_rule(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict verification kills a sandbox that applied fewer rules than requested."""
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC, egress_verification="strict")
    sdk = environment._driver.sdk
    create = sdk["Sandbox"].create

    async def drop_rules(image: Any, **kwargs: Any) -> Sandbox:
        """Create normally, but apply the policy without its first rule."""
        sandbox = await create(image, **kwargs)
        sandbox._policy = {**sandbox._policy, "egress": sandbox._policy["egress"][1:]}
        return sandbox

    monkeypatch.setattr(sdk["Sandbox"], "create", drop_rules)

    with pytest.raises(RuntimeError, match="did not apply 1 requested egress rule"):
        await environment._create_sandbox()

    assert server.live() == set()


def test_rejects_compose_tasks_and_missing_images(tmp_path: Path, server: Server) -> None:
    """Compose tasks and tasks without a prebuilt image are rejected at construction."""
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "docker-compose.yaml").write_text("services: {}\n")
    with pytest.raises(ValueError, match="docker-compose"):
        _environment(tmp_path, server, NetworkMode.PUBLIC)

    (tmp_path / "environment" / "docker-compose.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="prebuilt image"):
        NemoOpenSandboxEnvironment(
            environment_dir=tmp_path / "environment",
            environment_name="hello-world",
            session_id="s",
            trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
            task_env_config=EnvironmentConfig(),
            trusted_allowed_hosts=TRUSTED,
        )


async def test_start_prepares_logs_and_uploads_the_environment(tmp_path: Path, server: Server) -> None:
    """``start`` creates the log and work directories, opens them up as root, and uploads the environment directory with its file modes."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK, task={"workdir": "/app"})
    script = tmp_path / "environment" / "setup.sh"
    script.write_text("echo hi\n")
    script.chmod(0o755)

    await environment.start(force_build=False)

    sandbox = _sandbox(environment)
    assert {"/logs/agent", "/logs/verifier", "/logs/artifacts", "/app"} <= set(sandbox.files.tree)
    chmod = next(opts for command, opts in sandbox.commands.runs if command.startswith("chmod 777"))
    assert chmod["uid"] == 0
    assert chmod["working_directory"] == "/"
    assert sandbox.files.tree["/app/setup.sh"] == b"echo hi\n"
    assert sandbox.files.modes["/app/setup.sh"] == 755


async def test_stop_deletes_and_closes(tmp_path: Path, server: Server) -> None:
    """``stop(delete=True)`` kills the sandbox and releases the client."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)
    await environment.start(force_build=False)
    sandbox = _sandbox(environment)

    await environment.stop(delete=True)

    assert server.live() == set()
    assert sandbox.closed
    assert environment._sandbox is None


async def test_file_transfers_round_trip(tmp_path: Path, server: Server) -> None:
    """Files and directories upload and download intact, and an unreadable entry is skipped."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)
    await environment.start(force_build=False)
    source = tmp_path / "upload"
    (source / "nested").mkdir(parents=True)
    (source / "a.txt").write_text("a")
    (source / "nested" / "b.txt").write_text("b")

    await environment.upload_dir(source, "/data")
    await environment.upload_file(source / "a.txt", "/single/a.txt")
    files = _sandbox(environment).files
    files.tree["/data/broken-link"] = b""
    files.unreadable.add("/data/broken-link")
    await environment.download_dir("/data", tmp_path / "out")
    await environment.download_file("/single/a.txt", tmp_path / "one" / "a.txt")

    assert (tmp_path / "out" / "a.txt").read_text() == "a"
    assert (tmp_path / "out" / "nested" / "b.txt").read_text() == "b"
    assert not (tmp_path / "out" / "broken-link").exists()
    assert (tmp_path / "one" / "a.txt").read_text() == "a"


async def test_download_dir_asks_the_sandbox_for_untyped_entries(tmp_path: Path, server: Server) -> None:
    """Entries the search returns without a type are checked with ``test -d`` and ``test -f``."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)
    await environment.start(force_build=False)
    files = _sandbox(environment).files
    files.tree.update({"/data/sub": None, "/data/sub/c.txt": b"c"})
    files.untyped.update({"/data/sub", "/data/sub/c.txt"})

    await environment.download_dir("/data", tmp_path / "out")

    assert (tmp_path / "out" / "sub").is_dir()
    assert (tmp_path / "out" / "sub" / "c.txt").read_text() == "c"
    asked = [command for command, _ in _sandbox(environment).commands.runs if command.startswith("test -")]
    assert asked == ["test -d /data/sub", "test -d /data/sub/c.txt", "test -f /data/sub/c.txt"]


async def test_exec_falls_back_to_the_task_workdir_and_merges_env(tmp_path: Path, server: Server) -> None:
    """``exec`` defaults to the task's workdir, merges env, and returns stderr and the exit code."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK, task={"workdir": "/app"})
    await environment.start(force_build=False)
    commands = _sandbox(environment).commands
    commands.results["false"] = execution(stderr=["nope"], exit_code=1)

    result = await environment.exec("false", env={"EXTRA": "1"}, timeout_sec=5)
    await environment.exec("pwd", cwd="/tmp")

    assert (result.stderr, result.return_code) == ("nope", 1)
    opts = dict(commands.runs)
    assert opts["false"]["working_directory"] == "/app"
    assert opts["false"]["envs"]["EXTRA"] == "1"
    assert opts["pwd"]["working_directory"] == "/tmp"


async def _started(tmp_path: Path, server: Server, image_uid: int | None) -> NemoOpenSandboxEnvironment:
    """An environment holding a live mock sandbox whose image runs as ``image_uid``, without running ``start``."""
    server.image_uid = image_uid
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)
    environment._sandbox = Sandbox(server, "sbx-test", None)
    return environment


async def test_root_commands_run_as_the_image_user_on_a_non_root_image(tmp_path: Path, server: Server) -> None:
    """On a non-root image, root commands run as the image's user, and the image is probed only once."""
    environment = await _started(tmp_path, server, image_uid=10001)
    commands = _sandbox(environment).commands

    await environment.exec("tmux -V", user="root")
    await environment.exec("chmod 777 /logs/agent", user=0)

    assert commands.uids_sent("tmux -V") == [None]
    assert commands.uids_sent("chmod 777 /logs/agent") == [None]
    assert commands.probes() == 1


async def test_a_root_default_user_is_mapped_too(tmp_path: Path, server: Server) -> None:
    """A default user of ``root`` is mapped the same way as an explicit one."""
    environment = await _started(tmp_path, server, image_uid=10001)
    environment.default_user = "root"

    await environment.exec("mkdir -p /installed-agent")

    assert _sandbox(environment).commands.uids_sent("mkdir -p /installed-agent") == [None]


async def test_root_images_keep_uid_0(tmp_path: Path, server: Server) -> None:
    """On a root image, root commands keep uid 0."""
    environment = await _started(tmp_path, server, image_uid=0)

    await environment.exec("apt-get install -y curl", user="root")

    assert _sandbox(environment).commands.uids_sent("apt-get install -y curl") == [0]


async def test_non_root_requests_do_not_probe(tmp_path: Path, server: Server) -> None:
    """Commands that don't ask for root never trigger the image probe."""
    environment = await _started(tmp_path, server, image_uid=10001)
    commands = _sandbox(environment).commands

    await environment.exec("whoami")
    await environment.exec("whoami", user=1234)
    await environment.exec("whoami", user="agent")

    assert commands.uids_sent("whoami") == [None, 1234, None]
    assert commands.probes() == 0


async def test_a_failed_probe_keeps_uid_0(tmp_path: Path, server: Server) -> None:
    """When the image probe fails, root commands keep uid 0, and the probe isn't retried."""
    environment = await _started(tmp_path, server, image_uid=None)
    commands = _sandbox(environment).commands

    await environment.exec("tmux -V", user="root")
    await environment.exec("tmux -V", user="root")

    assert commands.uids_sent("tmux -V") == [0, 0]
    assert commands.probes() == 1


def _resource_sent(server: Server) -> dict[str, str]:
    return server.creates[-1]["resource"]


async def test_sandbox_disk_is_limited_to_the_tasks_storage(tmp_path: Path, server: Server) -> None:
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC, task={"storage_mb": 10240})

    await environment._create_sandbox()

    assert _resource_sent(server)["ephemeral-storage"] == "10240Mi"


async def test_disk_limit_is_capped_and_the_cap_is_recorded(tmp_path: Path, server: Server) -> None:
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC, task={"storage_mb": 1_024_000}, max_disk_mb=131072)

    sandbox = await environment._create_sandbox()

    assert _resource_sent(server)["ephemeral-storage"] == "131072Mi"
    record = json.loads((tmp_path / TRIAL / applied_egress_filename(sandbox.id)).read_text())
    assert (record["disk_requested_mb"], record["disk_limit_mb"]) == (1_024_000, 131072)


async def test_task_below_the_cap_keeps_its_own_limit(tmp_path: Path, server: Server) -> None:
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC, task={"storage_mb": 4096}, max_disk_mb=131072)

    await environment._create_sandbox()

    assert _resource_sent(server)["ephemeral-storage"] == "4096Mi"


@pytest.mark.parametrize(
    ("storage_mb", "kwargs"),
    [(None, {}), (10240, {"enforce_disk_limit": False})],
    ids=["task-sets-no-storage", "enforcement-off"],
)
async def test_no_disk_limit_is_sent_and_none_is_recorded(
    tmp_path: Path, server: Server, storage_mb: int | None, kwargs: dict[str, Any]
) -> None:
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC, task={"storage_mb": storage_mb}, **kwargs)

    sandbox = await environment._create_sandbox()

    assert "ephemeral-storage" not in _resource_sent(server)
    record = json.loads((tmp_path / TRIAL / applied_egress_filename(sandbox.id)).read_text())
    assert record["disk_limit_mb"] is None


async def test_cpu_and_memory_limits_still_reach_the_server(tmp_path: Path, server: Server) -> None:
    environment = _environment(
        tmp_path, server, NetworkMode.PUBLIC, task={"storage_mb": 10240, "cpus": 2, "memory_mb": 4096}
    )

    await environment._create_sandbox()

    resource = _resource_sent(server)
    assert resource["cpu"] == "2"
    assert resource["memory"] == "4096Mi"


async def test_a_retried_create_sends_the_same_disk_limit(tmp_path: Path, server: Server) -> None:
    server.create_failures.append(SandboxApiException("transient"))
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC, task={"storage_mb": 10240})

    await environment._create_sandbox()

    assert len(server.creates) == 2
    assert [create["resource"]["ephemeral-storage"] for create in server.creates] == ["10240Mi", "10240Mi"]


def _record(tmp_path: Path, sandbox_id: str) -> dict[str, Any]:
    return json.loads((tmp_path / TRIAL / applied_egress_filename(sandbox_id)).read_text())


@pytest.mark.parametrize("use_server_proxy", [False, True], ids=["direct", "server-proxy"])
async def test_records_the_pod_ip_and_the_window_it_was_held(
    tmp_path: Path, server: Server, use_server_proxy: bool
) -> None:
    """Gateway calls are attributed by source IP and time, so each record names its pod IP and lifetime window.

    The pod IP is asked of the server without ``use_server_proxy``; a client that proxies execd would
    otherwise get a server route back and no sandbox in a taskset run could be attributed.
    """
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC, use_server_proxy=use_server_proxy)
    await environment.start(force_build=False)
    sandbox_id = _sandbox(environment).id

    record = _record(tmp_path, sandbox_id)
    assert record["ip"] == server.pod_ip(sandbox_id)
    assert record["stopped_at"] is None
    assert [request.url.params.get("use_server_proxy") for request in server.endpoint_requests] == [None]

    await environment.stop(delete=True)

    record = _record(tmp_path, sandbox_id)
    started_at = datetime.fromisoformat(record["started_at"])
    stopped_at = datetime.fromisoformat(record["stopped_at"])
    assert started_at.tzinfo is not None
    assert started_at <= stopped_at


async def test_the_recorded_window_lies_inside_the_pods_lifetime(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``started_at`` is taken once create returns and ``stopped_at`` before the kill is sent.

    While a pod exists no other pod holds its IP, so windows inside pod lifetimes never overlap
    for one IP; a window that spilled past either end could claim another sandbox's calls.
    """
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)
    real_create = environment._driver.create
    created_at: list[datetime] = []

    async def create(*args: Any, **kwargs: Any) -> Any:
        sandbox = await real_create(*args, **kwargs)
        created_at.append(datetime.now(UTC))
        return sandbox

    monkeypatch.setattr(environment._driver, "create", create)
    await environment.start(force_build=False)
    sandbox = _sandbox(environment)
    real_kill = sandbox.kill
    record_at_kill: list[dict[str, Any]] = []

    async def kill() -> None:
        record_at_kill.append(_record(tmp_path, sandbox.id))
        killed_at = datetime.now(UTC)
        await real_kill()
        assert datetime.fromisoformat(record_at_kill[0]["stopped_at"]) <= killed_at

    monkeypatch.setattr(sandbox, "kill", kill)
    await environment.stop(delete=True)

    assert datetime.fromisoformat(record_at_kill[0]["started_at"]) >= created_at[0]
    assert record_at_kill[0]["stopped_at"] is not None
    assert server.killed == [sandbox.id]


async def test_stopped_at_is_recorded_when_the_sandbox_is_kept(tmp_path: Path, server: Server) -> None:
    """``stop(delete=False)`` still ends the window; the sandbox's calls after it go unattributed, not misattributed."""
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)
    await environment.start(force_build=False)
    sandbox_id = _sandbox(environment).id

    await environment.stop(delete=False)

    assert _record(tmp_path, sandbox_id)["stopped_at"] is not None
    assert server.killed == []


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(200, json={"endpoint": "gateway.example/sandboxes/sbx-1/port/44772"}),
        httpx.Response(200, json={"endpoint": "127.0.0.1:32768"}),
        httpx.Response(404, json={"detail": "Pod IP is not yet available"}),
        httpx.Response(200, text="not json"),
        httpx.ConnectError("refused"),
    ],
    ids=["gateway-route", "host-mapped-port", "not-found", "bad-body", "unreachable"],
)
async def test_a_sandbox_without_a_readable_pod_ip_is_recorded_without_one(
    tmp_path: Path, server: Server, caplog: pytest.LogCaptureFixture, answer: httpx.Response | Exception
) -> None:
    """Anything but ``<IP>:44772`` records no IP, so that sandbox's calls go unattributed rather than misattributed."""
    server.endpoint_answer = answer
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)

    sandbox = await environment._create_sandbox()

    record = _record(tmp_path, sandbox.id)
    assert record["ip"] is None
    assert record["started_at"] is not None
    assert "recorded without" in caplog.text
    assert server.live() == {sandbox.id}


async def test_a_create_cancelled_while_reading_the_pod_ip_kills_the_sandbox(tmp_path: Path, server: Server) -> None:
    """The sandbox isn't the environment's until create returns, so a cancellation before that must not leak it."""
    server.endpoint_answer = asyncio.CancelledError()
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)

    with pytest.raises(asyncio.CancelledError):
        await environment._create_sandbox()

    assert server.live() == set()


async def test_stop_before_start_records_nothing(tmp_path: Path, server: Server) -> None:
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)

    await environment.stop(delete=True)

    assert list((tmp_path / TRIAL).glob(APPLIED_EGRESS_GLOB)) == []


async def test_the_pod_ip_lookup_stops_retrying_at_its_budget(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The IP is optional, so an unresponsive server can't hold the create for every retry's full timeout."""
    monkeypatch.setattr(environment_module, "SANDBOX_IP_LOOKUP_BUDGET_SEC", 0)
    server.endpoint_answer = httpx.ReadTimeout("no answer")
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)

    sandbox = await environment._create_sandbox()

    assert len(server.endpoint_requests) == 1
    assert _record(tmp_path, sandbox.id)["ip"] is None


async def test_a_failed_rewrite_leaves_the_previous_record_whole(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that dies after emptying its file (a full disk, say) must not cost the sandbox its IP and start."""
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)
    await environment.start(force_build=False)
    sandbox_id = _sandbox(environment).id
    before = _record(tmp_path, sandbox_id)

    def dies_mid_write(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        with open(self, "w") as file:
            file.write(data[:10])
        raise OSError("No space left on device")

    monkeypatch.setattr(Path, "write_text", dies_mid_write)
    await environment.stop(delete=True)
    monkeypatch.undo()

    assert _record(tmp_path, sandbox_id) == before
    assert [path.name for path in (tmp_path / TRIAL).iterdir()] == [applied_egress_filename(sandbox_id)]


async def test_a_slowly_answered_pod_ip_lookup_still_stops_at_its_budget(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server dripping its answer resets the client's read timeout per chunk, so only a total bound stops it."""
    monkeypatch.setattr(environment_module, "SANDBOX_IP_LOOKUP_BUDGET_SEC", 0.1)
    environment = _environment(tmp_path, server, NetworkMode.PUBLIC)

    class _Dripping:
        async def get(self, path: str, **kwargs: Any) -> httpx.Response:
            await asyncio.sleep(30)
            raise AssertionError("the budget should have stopped the lookup")

    monkeypatch.setattr(environment, "_server", lambda: _Dripping())

    started = time.monotonic()
    sandbox = await environment._create_sandbox()

    assert time.monotonic() - started < 5
    assert _record(tmp_path, sandbox.id)["ip"] is None
    assert server.live() == {sandbox.id}


def _terminations(tmp_path: Path) -> list[dict[str, Any]]:
    """Every termination record in the trial directory, by file name."""
    return [json.loads(path.read_text()) for path in sorted((tmp_path / TRIAL).glob(TERMINATION_GLOB))]


async def test_stop_records_nothing_for_a_sandbox_that_is_still_running(tmp_path: Path, server: Server) -> None:
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)
    await environment.start(force_build=False)

    await environment.stop(delete=True)

    assert _terminations(tmp_path) == []
    assert server.live() == set()


async def test_stop_records_an_eviction_before_it_kills_the_sandbox(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server 1.1 reports an evicted sandbox as Pending; the cause comes from its warning event."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)
    await environment.start(force_build=False)
    sandbox = _sandbox(environment)
    server.states = ["Pending"]
    server.status_reason, server.status_message = "CREATING", "Sandbox is being created"
    server.events_content = (
        "[2026-10-30 20:05:07+00:00] Warning  Evicted  Pod ephemeral local storage usage exceeds the total limit of containers 200Mi.\n"
        "[2026-10-30 20:05:07+00:00] Normal   Killing  Stopping container sandbox\n"
    )
    real_kill = sandbox.kill
    recorded_at_kill: list[list[dict[str, Any]]] = []

    async def kill() -> None:
        recorded_at_kill.append(_terminations(tmp_path))
        await real_kill()

    monkeypatch.setattr(sandbox, "kill", kill)
    await environment.stop(delete=True)

    [termination] = _terminations(tmp_path)
    assert recorded_at_kill == [[termination]]
    assert (termination["backend"], termination["sandbox_id"], termination["role"]) == (
        "opensandbox",
        sandbox.id,
        "agent",
    )
    assert termination["cause"] == "evicted"
    assert termination["evidence"] == {
        "state": "Pending",
        "reason": "CREATING",
        "message": "Sandbox is being created",
        "events": ["Evicted: Pod ephemeral local storage usage exceeds the total limit of containers 200Mi."],
    }
    assert server.live() == set()


async def test_a_separate_verifiers_sandbox_is_recorded_as_the_verifiers(tmp_path: Path, server: Server) -> None:
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK, session_id=f"{TRIAL}__verifier__tests")
    await environment.start(force_build=False)
    server.states = ["Failed"]

    await environment.stop(delete=True)

    [termination] = _terminations(tmp_path)
    assert (termination["role"], termination["cause"]) == ("verifier", "unknown")
    assert termination["detail"] == "the sandbox stopped; server reported Failed: image pull failed"


async def test_a_sandbox_gone_before_its_lifetime_ended_was_deleted(tmp_path: Path, server: Server) -> None:
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)
    await environment.start(force_build=False)
    server.gone.add(_sandbox(environment).id)

    await environment.stop(delete=True)

    [termination] = _terminations(tmp_path)
    assert (termination["cause"], termination["evidence"]) == ("deleted", {})


async def test_a_sandbox_gone_after_its_lifetime_reached_it(tmp_path: Path, server: Server) -> None:
    """The server deletes a sandbox at the end of its lifetime, so a 404 after then is the lifetime, not a deletion."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK, sandbox_timeout_sec=3600)
    await environment.start(force_build=False)
    assert environment._sandbox_ends_at is not None
    environment._sandbox_ends_at -= timedelta(hours=2)
    server.gone.add(_sandbox(environment).id)

    await environment.stop(delete=True)

    [termination] = _terminations(tmp_path)
    assert termination["cause"] == "lifetime_reached"


async def test_the_lifetime_ends_at_the_run_expiry_when_that_comes_first(tmp_path: Path, server: Server) -> None:
    expires_at = datetime.now(UTC) + timedelta(minutes=5)
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK, sandbox_expires_at=expires_at.isoformat())

    await environment.start(force_build=False)

    assert environment._sandbox_ends_at == expires_at


async def test_unreadable_events_still_record_the_stop(tmp_path: Path, server: Server) -> None:
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)
    await environment.start(force_build=False)
    server.states = ["Pending"]
    server.events_content = None

    await environment.stop(delete=True)

    [termination] = _terminations(tmp_path)
    assert (termination["cause"], termination["evidence"]["events"]) == ("unknown", [])


async def test_a_failed_status_read_records_nothing_and_still_stops(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A status read that fails says nothing about the sandbox, so nothing is recorded and the kill still goes out."""
    environment = _environment(tmp_path, server, NetworkMode.NO_NETWORK)
    await environment.start(force_build=False)
    monkeypatch.setattr(
        server, "sandbox_route", lambda request: httpx.Response(503) if request.method == "GET" else None
    )

    await environment.stop(delete=True)

    assert _terminations(tmp_path) == []
    assert "Could not check whether sandbox" in caplog.text
    assert server.live() == set()
