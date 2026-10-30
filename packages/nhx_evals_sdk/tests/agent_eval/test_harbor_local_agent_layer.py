# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Docker agent layers: one install per environment image and agent, built once across runs on the machine.

Docker itself is faked: these tests check what is asked of it, not Docker. The e2e suite runs the real thing.
"""

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor", reason="local agent layers build on Harbor's Docker environment")

from harbor.environments.docker.compose_env import ComposeInfraEnvVars
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from nhx_evals_sdk.agent_eval.runtimes.harbor import local_agent_layer
from nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer import (
    AGENT_LAYER_DOCKER_ENVIRONMENT,
    AGENT_LAYER_KWARG,
    AGENT_LAYERS_ENV,
    AgentInstall,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.local_agent_layer import (
    LAYER_LABEL,
    AgentLayerDockerEnvironment,
    ContainerExecEnvironment,
    build_agent_layer,
    ensure_agent_layer,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import _with_agent_layers

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="no flock on Windows")

_INSTALL = AgentInstall(agent="harbor:codex", inputs={"version": "0.162.1"}, harbor_version="0.24.0")
_AGENT = AgentConfig(name="codex", kwargs={"version": "0.162.1"})


class _Docker:
    """The images on a fake Docker host, and every build of a layer."""

    def __init__(self, images: set[str] | None = None) -> None:
        self.images = images if images is not None else {"task-image"}
        self.builds: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def exists(image: str) -> bool:
            return image in self.images

        async def image_id(image: str) -> str:
            return f"sha256:id-of-{image}"

        async def build(image: str, tag: str, install: AgentInstall, agent: AgentConfig, *, user: Any) -> None:
            await asyncio.sleep(0.05)  # long enough for a concurrent trial to start waiting
            self.builds.append(tag)
            self.images.add(tag)

        monkeypatch.setattr(local_agent_layer, "_image_exists", exists)
        monkeypatch.setattr(local_agent_layer, "_image_id", image_id)
        monkeypatch.setattr(local_agent_layer, "build_agent_layer", build)
        monkeypatch.setattr(local_agent_layer, "LOCK_POLL_INTERVAL_SEC", 0.01)


async def _ensure(tmp_path: Path, *, user: str | None = None, install: AgentInstall = _INSTALL) -> str:
    return await ensure_agent_layer("task-image", install, _AGENT, user=user, lock_dir=tmp_path / "locks")


async def test_the_first_trial_builds_the_layer_and_later_ones_reuse_it(tmp_path, monkeypatch) -> None:
    docker = _Docker()
    docker.install(monkeypatch)

    first = await _ensure(tmp_path)
    second = await _ensure(tmp_path)

    assert first == second and first.startswith("nemo-agent-layer:")
    assert docker.builds == [first]


async def test_concurrent_trials_build_one_layer(tmp_path, monkeypatch) -> None:
    docker = _Docker()
    docker.install(monkeypatch)

    tags = await asyncio.gather(*(_ensure(tmp_path) for _ in range(5)))

    assert len(set(tags)) == 1 and docker.builds == [tags[0]]


async def test_another_agent_user_or_install_gets_its_own_layer(tmp_path, monkeypatch) -> None:
    docker = _Docker()
    docker.install(monkeypatch)
    other_version = AgentInstall(agent="harbor:codex", inputs={"version": "0.163.0"}, harbor_version="0.24.0")

    tags = {
        await _ensure(tmp_path),
        await _ensure(tmp_path, user="agent"),
        await _ensure(tmp_path, install=other_version),
    }

    assert len(tags) == 3 and len(docker.builds) == 3


async def test_a_layer_another_run_is_building_is_waited_for_not_rebuilt(tmp_path, monkeypatch) -> None:
    """Another process holds the layer's lock while it builds; this trial waits, then uses what that one built."""
    docker = _Docker()
    docker.install(monkeypatch)
    tag = await _ensure(tmp_path)  # learn the key's tag and lock file
    docker.images.discard(tag)
    docker.builds.clear()
    lock = tmp_path / "locks" / f"{tag.rpartition(':')[2]}.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import fcntl, sys
                handle = open({str(lock)!r}, "ab")
                fcntl.flock(handle, fcntl.LOCK_EX)
                print("held", flush=True)
                sys.stdin.read()
            """),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        waiting = asyncio.ensure_future(_ensure(tmp_path))
        await asyncio.sleep(0.2)
        assert not waiting.done()
        docker.images.add(tag)  # the other run finished its build
    finally:
        holder.kill()
        holder.wait()

    assert await asyncio.wait_for(waiting, timeout=5) == tag
    assert docker.builds == []


async def test_a_missing_task_image_is_pulled_first(tmp_path, monkeypatch) -> None:
    docker = _Docker(images=set())
    docker.install(monkeypatch)
    pulled: list[tuple[str, ...]] = []

    async def fake_docker(*args: str, check: bool = True) -> str:
        pulled.append(args)
        docker.images.add(args[1])
        return ""

    monkeypatch.setattr(local_agent_layer, "_docker", fake_docker)

    await _ensure(tmp_path)

    assert pulled == [("pull", "task-image")]


async def test_a_layer_is_the_setup_of_a_bare_container_committed_with_its_cmd_restored(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []
    set_up: list[Any] = []

    async def fake_docker(*args: str, check: bool = True) -> str:
        calls.append(args)
        if args[:2] == ("image", "inspect"):
            labels = {"com.docker.compose.project": "hello-world__abc__env", "org.opencontainers.image.title": "t"}
            return json.dumps({"Cmd": ["bash"], "Entrypoint": None, "Labels": labels})
        return ""

    class _Agent:
        async def setup(self, environment: Any) -> None:
            set_up.append(environment)

    monkeypatch.setattr(local_agent_layer, "_docker", fake_docker)
    monkeypatch.setattr(local_agent_layer.AgentFactory, "create_agent_from_config", lambda *a, **k: _Agent())

    await build_agent_layer("task-image", "nemo-agent-layer:abc", _INSTALL, _AGENT, user="agent")

    run = next(call for call in calls if call[0] == "run")
    assert run[-4:] == ("task-image", "sh", "-c", "sleep infinity")
    (environment,) = set_up
    assert isinstance(environment, ContainerExecEnvironment) and environment.default_user == "agent"
    commit = next(call for call in calls if call[0] == "commit")
    changes = [commit[i + 1] for i, arg in enumerate(commit) if arg == "--change"]
    assert 'CMD ["bash"]' in changes
    # The trial's compose project label is blanked, or that trial's `down --rmi local` would delete the layer.
    assert 'LABEL "com.docker.compose.project"=""' in changes
    assert not any("org.opencontainers" in change for change in changes)
    assert f'LABEL "{LAYER_LABEL}"="abc"' in changes
    assert commit[-1] == "nemo-agent-layer:abc"
    assert calls[-1][:1] == ("stop",)


async def test_a_failed_setup_still_stops_the_build_container(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_docker(*args: str, check: bool = True) -> str:
        calls.append(args)
        return "{}" if args[:2] == ("image", "inspect") else ""

    class _Agent:
        async def setup(self, environment: Any) -> None:
            raise RuntimeError("npm unreachable")

    monkeypatch.setattr(local_agent_layer, "_docker", fake_docker)
    monkeypatch.setattr(local_agent_layer.AgentFactory, "create_agent_from_config", lambda *a, **k: _Agent())

    with pytest.raises(RuntimeError, match="npm unreachable"):
        await build_agent_layer("task-image", "nemo-agent-layer:abc", _INSTALL, _AGENT, user=None)
    assert not any(call[0] == "commit" for call in calls)
    assert calls[-1][:1] == ("stop",)


async def test_container_exec_runs_as_the_default_user_unless_told_otherwise(monkeypatch) -> None:
    ran: list[tuple[str, ...]] = []

    class _Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"out", b""

    async def create(*args: str, **kwargs: Any) -> _Process:
        ran.append(args)
        return _Process()

    monkeypatch.setattr(local_agent_layer.asyncio, "create_subprocess_exec", create)
    environment = ContainerExecEnvironment("box", default_user="agent")

    result = await environment.exec("npm -v", cwd="/app", env={"A": "1"})
    await environment.exec("apt-get update", user="root")

    assert result.stdout == "out" and result.return_code == 0
    assert ran[0] == (
        "docker",
        "exec",
        "--user",
        "agent",
        "--workdir",
        "/app",
        "--env",
        "A=1",
        "box",
        "bash",
        "-c",
        "npm -v",
    )
    assert ran[1][:4] == ("docker", "exec", "--user", "root")


def _environment(tmp_path: Path, agent: dict[str, Any], *, compose: bool = False) -> AgentLayerDockerEnvironment:
    """The environment's state ``_validate_image_os`` reads, without starting Harbor's Docker machinery."""
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "task.toml").write_text('[agent]\nuser = "agent"\n')
    if compose:
        (task_dir / "environment" / "docker-compose.yaml").write_text("services: {}\n")
    environment = object.__new__(AgentLayerDockerEnvironment)
    environment._agent = agent
    environment._is_windows_container = False
    environment.environment_dir = task_dir / "environment"
    environment._env_vars = ComposeInfraEnvVars(main_image_name="hb__abc", context_dir=str(task_dir / "environment"))
    environment._use_prebuilt = False
    environment._before_layer = None
    environment.session_id = "Hello-World__abc"
    environment.__dict__["environment_id"] = "env-content-hash"  # Harbor's cached_property, over environment/
    return environment


async def _validated(environment: AgentLayerDockerEnvironment, monkeypatch, layer: str | Exception) -> list[Any]:
    asked: list[Any] = []

    async def parent_validate(self: Any, image_name: str) -> None:
        return None

    async def ensure(
        image: str, install: AgentInstall, agent: AgentConfig, *, user: Any, lock_dir: Path, base_key: str | None
    ) -> str:
        asked.append((image, install.agent, user, base_key))
        if isinstance(layer, Exception):
            raise layer
        return layer

    monkeypatch.setattr(local_agent_layer.DockerEnvironment, "_validate_image_os", parent_validate)
    monkeypatch.setattr(local_agent_layer, "ensure_agent_layer", ensure)
    await environment._validate_image_os("hb__task")
    return asked


_CODEX = {"name": "codex", "import_path": None, "kwargs": {"version": "0.162.1"}}


async def test_the_trial_starts_from_its_agent_layer(tmp_path, monkeypatch) -> None:
    environment = _environment(tmp_path, _CODEX)

    asked = await _validated(environment, monkeypatch, "nemo-agent-layer:abc")

    # The image compose built, keyed by environment/'s contents: compose labels each trial's build differently.
    assert asked == [("hello-world__abc-main", "harbor:codex", "agent", "environment:env-content-hash")]
    assert environment._use_prebuilt and environment._env_vars.prebuilt_image_name == "nemo-agent-layer:abc"


async def test_a_layer_that_cant_be_built_leaves_the_trial_on_its_image(tmp_path, monkeypatch) -> None:
    environment = _environment(tmp_path, _CODEX)

    await _validated(environment, monkeypatch, RuntimeError("docker commit failed"))

    assert not environment._use_prebuilt and environment._env_vars.prebuilt_image_name is None


@pytest.mark.parametrize(
    ("agent", "compose"),
    [({"name": "oracle", "import_path": None, "kwargs": {}}, False), (_CODEX, True)],
    ids=["agent-installs-nothing", "task-compose-file"],
)
async def test_no_layer_when_it_cant_apply(tmp_path, monkeypatch, agent: dict[str, Any], compose: bool) -> None:
    environment = _environment(tmp_path, agent, compose=compose)

    assert await _validated(environment, monkeypatch, "nemo-agent-layer:abc") == []
    assert not environment._use_prebuilt


def test_the_default_docker_environment_reuses_installs_of_an_installed_agent(monkeypatch) -> None:
    monkeypatch.delenv(AGENT_LAYERS_ENV, raising=False)
    environment = EnvironmentConfig(extra_docker_compose=[])

    switched = _with_agent_layers(environment, _AGENT, agent_dir=None)

    assert switched.import_path == AGENT_LAYER_DOCKER_ENVIRONMENT
    assert switched.kwargs[AGENT_LAYER_KWARG] == {
        "name": "codex",
        "import_path": None,
        "kwargs": {"version": "0.162.1"},
    }


@pytest.mark.parametrize(
    ("environment", "agent", "env"),
    [
        (EnvironmentConfig(), AgentConfig(name="oracle"), None),
        (EnvironmentConfig(import_path="my.env:Env"), _AGENT, None),
        (EnvironmentConfig(), _AGENT, "0"),
    ],
    ids=["agent-installs-nothing", "custom-environment", "turned-off"],
)
def test_other_environments_and_agents_are_left_alone(monkeypatch, environment, agent, env: str | None) -> None:
    if env is None:
        monkeypatch.delenv(AGENT_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(AGENT_LAYERS_ENV, env)

    assert _with_agent_layers(environment, agent, agent_dir=None) is environment


async def test_a_prebuilt_task_image_is_keyed_by_its_image(tmp_path, monkeypatch) -> None:
    environment = _environment(tmp_path, _CODEX)
    environment._use_prebuilt = True

    asked = await _validated(environment, monkeypatch, "nemo-agent-layer:abc")

    assert asked == [("hb__task", "harbor:codex", "agent", None)]


@pytest.mark.parametrize(
    ("prebuilt", "image"), [(False, None), (True, "ghcr.io/org/task:1")], ids=["built", "prebuilt"]
)
async def test_teardown_removes_the_trials_own_image_not_the_shared_layer(
    tmp_path, monkeypatch, prebuilt: bool, image: str | None
) -> None:
    """Harbor's ``down --rmi local`` removes the service's image, so the service is pointed back at its own first."""
    environment = _environment(tmp_path, _CODEX)
    environment._use_prebuilt, environment._env_vars.prebuilt_image_name = prebuilt, image
    await _validated(environment, monkeypatch, "nemo-agent-layer:abc")
    assert environment._env_vars.prebuilt_image_name == "nemo-agent-layer:abc"
    stopped: list[tuple[bool, str | None]] = []

    async def parent_stop(self: Any, delete: bool) -> None:
        stopped.append((self._use_prebuilt, self._env_vars.prebuilt_image_name))

    monkeypatch.setattr(local_agent_layer.DockerEnvironment, "stop", parent_stop)
    await environment.stop(delete=True)

    assert stopped == [(prebuilt, image)]
