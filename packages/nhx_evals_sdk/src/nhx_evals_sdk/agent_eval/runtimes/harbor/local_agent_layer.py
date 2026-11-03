# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent layers for local Docker trials: install the agent once per environment image, then start trials from it.

:class:`AgentLayerDockerEnvironment` is Harbor's Docker environment with one change. Once the trial's environment
image is built or pulled, and before its container starts, the trial switches to that image's *agent layer*: the
image with the agent already installed, tagged ``nemo-agent-layer:<key>``. The first trial that needs a layer
builds it:
- start a throwaway container from the environment image;
- run the agent's own ``setup()`` in it, as the task's agent user, the way a trial would, but with nothing of a
  trial (no healthcheck, task files, skills or credentials);
- ``docker commit`` the result, restoring the image's ``ENTRYPOINT`` and ``CMD``.

Trials of the same key across runs on the machine wait for that build (an exclusive ``flock`` per key) and reuse
it. A layer that can't be built leaves the trial on its environment image, where setup installs as before.

Set :data:`~nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer.AGENT_LAYERS_ENV` to ``0`` to turn layers off. List layers with
``docker images --filter label=nemo.agent-layer``; remove them with
``docker image prune --all --filter label=nemo.agent-layer``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, cast, override

from harbor.agents.factory import AgentFactory
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.docker.docker import DockerEnvironment, _sanitize_docker_compose_project_name
from harbor.models.task.config import TaskConfig
from harbor.models.trial.config import AgentConfig
from nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer import AgentInstall, agent_install

if sys.platform != "win32":
    import fcntl

logger = logging.getLogger(__name__)

LAYER_REPOSITORY = "nemo-agent-layer"
LAYER_LABEL = "nemo.agent-layer"
LOCK_POLL_INTERVAL_SEC = 0.5
#: How long a failed or cancelled layer build stops other trials from trying the same build: they install the agent in setup
#: instead, at once, rather than each repeating the failure in turn while holding the build's lock.
FAILED_BUILD_COOLDOWN_SEC = 600
#: The kwarg holding a Fabric agent's config bundle, a directory its setup uploads; never part of a layer.
FABRIC_CONFIG_BUNDLE_KWARG = "fabric_config_bundle"
_COMPOSE_LABEL_PREFIX = "com.docker.compose."


class AgentLayerDockerEnvironment(DockerEnvironment):
    """Harbor's Docker environment, starting each trial from its environment image's agent layer.

    A built image is tagged per trial (compose's ``<project>-main``) and compose labels it per project, so its ID
    differs between trials of the same ``environment/``. Its layer is keyed by :func:`environment_digest` of
    ``environment/`` instead; a prebuilt task image by its image ID, which every trial shares.
    """

    def __init__(self, *args: Any, agent_layer: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._agent = agent_layer or {}
        #: How the service started before switching to a layer, ``(use_prebuilt, prebuilt_image_name)``, or None.
        self._before_layer: tuple[bool, str | None] | None = None

    @override
    async def _validate_image_os(self, image_name: str) -> None:
        """Harbor calls this with the image the container will run, once it exists; switch to its agent layer."""
        await super()._validate_image_os(image_name)
        built = not self._use_prebuilt
        if built:
            digest = await asyncio.to_thread(environment_digest, self.environment_dir)
            layer = await self._agent_layer(self._built_image, base_key=f"environment:{digest}")
        else:
            layer = await self._agent_layer(image_name, base_key=None)
        if layer is not None:
            self._before_layer = (self._use_prebuilt, self._env_vars.prebuilt_image_name)
            self._env_vars.prebuilt_image_name = layer
            self._use_prebuilt = True

    @override
    async def stop(self, delete: bool) -> None:
        """Tear down as Harbor would have without the layer, so ``down --rmi local`` still removes this trial's build."""
        if self._before_layer is not None:
            self._use_prebuilt, self._env_vars.prebuilt_image_name = self._before_layer
        await super().stop(delete)

    @property
    def _built_image(self) -> str:
        """The image ``docker compose build`` made: Harbor's build file names none, so compose names it per project."""
        return f"{_sanitize_docker_compose_project_name(self.session_id)}-main"

    async def _agent_layer(self, image_name: str, *, base_key: str | None) -> str | None:
        if self._is_windows_container or self._environment_docker_compose_path.exists():
            return None  # A task's own compose file may define `main` itself.
        kwargs = self._agent.get("kwargs") or {}
        env = self._agent.get("env") or {}
        install = agent_install(
            agent_name=self._agent.get("name"),
            agent_import_path=self._agent.get("import_path"),
            agent_dir=None,
            agent_kwargs=kwargs,
            agent_env=env,
        )
        if install is None:
            return None
        try:
            user = _agent_user(self.environment_dir)
            return await ensure_agent_layer(
                image_name,
                install,
                # The Fabric config bundle is the trial's, not the install's: setup uploads it in every trial.
                AgentConfig(
                    name=self._agent.get("name"),
                    import_path=self._agent.get("import_path"),
                    kwargs={name: value for name, value in kwargs.items() if name != FABRIC_CONFIG_BUNDLE_KWARG},
                    env=env,
                ),
                user=user,
                lock_dir=default_lock_dir(),
                base_key=base_key,
            )
        except AgentLayerBuildFailedError as exc:
            logger.info("%s; the trial installs its agent in setup", exc)
            return None
        except Exception:
            logger.exception("Could not build the agent layer of %s; the trial installs its agent in setup", image_name)
            return None


def environment_digest(environment_dir: Path) -> str:
    """A digest of everything a build of ``environment_dir`` can see: each path, mode, and file's contents or link's target.

    Not Harbor's content hash, which leaves modes and symlinks out: a script made executable, or a link pointed
    elsewhere, builds a different image and must get a different layer.
    """
    digest = hashlib.sha256()
    for directory, names, files in os.walk(environment_dir):  # links to directories are listed, not followed
        names.sort()
        for name in sorted([*names, *files]):
            path = Path(directory, name)
            info = path.lstat()
            relative = path.relative_to(environment_dir).as_posix()
            if stat.S_ISLNK(info.st_mode):
                entry = f"link {relative} {os.readlink(path)}"
            elif stat.S_ISDIR(info.st_mode):
                entry = f"dir {relative} {stat.S_IMODE(info.st_mode):o}"
            else:
                contents = hashlib.sha256(path.read_bytes()).hexdigest()
                entry = f"file {relative} {stat.S_IMODE(info.st_mode):o} {contents}"
            digest.update(entry.encode() + b"\0")
    return digest.hexdigest()


def default_lock_dir() -> Path:
    return Path(tempfile.gettempdir()) / "nemo-agent-layers"


def _agent_user(environment_dir: Path) -> str | int | None:
    """The user the task runs its agent as: the layer is installed as that user, since installs write into $HOME."""
    task_toml = environment_dir.parent / "task.toml"
    if not task_toml.is_file():
        return None
    return TaskConfig.model_validate_toml(task_toml.read_text()).agent.user


async def ensure_agent_layer(
    image: str,
    install: AgentInstall,
    agent: AgentConfig,
    *,
    user: str | int | None,
    lock_dir: Path,
    base_key: str | None = None,
) -> str:
    """The tag of ``install``'s layer on ``image``, building it unless this machine already has it.

    The layer is keyed by ``base_key`` when given (what the image was built from), else by ``image``'s ID.
    """
    if not await _image_exists(image):
        await _docker("pull", image)  # a task's prebuilt image, which compose would otherwise pull on `up`
    base = base_key if base_key is not None else await _image_id(image)
    key = install.layer_key(f"{base}|user={user}")
    tag = f"{LAYER_REPOSITORY}:{key}"
    if await _image_exists(tag):
        return tag
    lock_dir.mkdir(parents=True, exist_ok=True)
    failed = lock_dir / f"{key}.failed"
    with (lock_dir / f"{key}.lock").open("ab") as lock:
        await _lock(lock, key)
        # Another trial, here or in another run, may have built it while this one waited, or failed to.
        if await _image_exists(tag):
            return tag
        if _failed_recently(failed):
            raise AgentLayerBuildFailedError(
                f"agent layer {tag} failed to build in the last {FAILED_BUILD_COOLDOWN_SEC} s"
            )
        try:
            await build_agent_layer(image, tag, install, agent, user=user)
        except BaseException:  # a build cancelled by the trial's timeout counts too: the next one would take as long
            failed.touch()
            raise
        failed.unlink(missing_ok=True)
    return tag


class AgentLayerBuildFailedError(RuntimeError):
    """The layer's build failed recently, so this trial installs its agent in setup instead of trying again."""


def _failed_recently(marker: Path) -> bool:
    try:
        return time.time() - marker.stat().st_mtime < FAILED_BUILD_COOLDOWN_SEC
    except FileNotFoundError:
        return False


async def _lock(lock: Any, key: str) -> None:
    if sys.platform == "win32":
        return
    waiting = False
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if not waiting:
                logger.info("Waiting for another trial to build agent layer %s", key)
                waiting = True
            await asyncio.sleep(LOCK_POLL_INTERVAL_SEC)


async def build_agent_layer(
    image: str, tag: str, install: AgentInstall, agent: AgentConfig, *, user: str | int | None
) -> None:
    """Install ``agent`` in a throwaway container from ``image`` and commit it as ``tag``."""
    # Unique: a failed build's container may still be on its way out when the next trial tries.
    container = f"nemo-agent-layer-build-{tag.rpartition(':')[2]}-{uuid.uuid4().hex[:8]}"
    config = json.loads(await _docker("image", "inspect", "--format", "{{json .Config}}", image))
    logger.info("Building agent layer %s (%s) on %s", tag, install.agent, image)
    # The image's ENTRYPOINT is overridden too: a task's entrypoint may initialize state each trial must make for
    # itself, which a layer would otherwise share.
    await _docker("run", "--detach", "--rm", "--name", container, "--entrypoint", "sh", image, "-c", "sleep infinity")
    try:
        with tempfile.TemporaryDirectory(prefix="nemo-agent-layer-") as logs:
            harbor_agent = AgentFactory.create_agent_from_config(agent, logs_dir=Path(logs))
            # Agents' setup() uses only exec, upload_file/upload_dir and default_user, which is all a bare
            # container offers: no trial, task directory or lifecycle to build a real BaseEnvironment from.
            environment = cast(BaseEnvironment, ContainerExecEnvironment(container, default_user=user))
            await harbor_agent.setup(environment=environment)
        labels = {LAYER_LABEL: tag.rpartition(":")[2], f"{LAYER_LABEL}.agent": install.agent, **_input_labels(install)}
        # A compose-built image carries its trial's project label, and `down --rmi local` deletes every image that
        # names its project: blank those, or that trial's teardown would delete the shared layer.
        compose = {name: "" for name in config.get("Labels") or {} if name.startswith(_COMPOSE_LABEL_PREFIX)}
        changes = [
            f"ENTRYPOINT {json.dumps(config.get('Entrypoint') or [])}",
            f"CMD {json.dumps(config.get('Cmd') or [])}",
        ]
        changes += [f"LABEL {json.dumps(name)}={json.dumps(value)}" for name, value in {**compose, **labels}.items()]
        await _docker("commit", *[arg for change in changes for arg in ("--change", change)], container, tag)
    finally:
        await _docker("stop", "--time", "1", container, check=False)
    logger.info("Built agent layer %s", tag)


def _input_labels(install: AgentInstall) -> dict[str, str]:
    return {f"{LAYER_LABEL}.{name}": value for name, value in install.inputs.items()} | {
        f"{LAYER_LABEL}.harbor": install.harbor_version
    }


class ContainerExecEnvironment:
    """What a Harbor agent's ``setup()`` uses of its environment, over ``docker exec`` into one container."""

    session_id = "agent-layer-build"

    def __init__(self, container: str, *, default_user: str | int | None) -> None:
        self.container = container
        self.default_user = default_user

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        args = ["exec"]
        if (user := user if user is not None else self.default_user) is not None:
            args += ["--user", str(user)]
        if cwd is not None:
            args += ["--workdir", cwd]
        for name, value in (env or {}).items():
            args += ["--env", f"{name}={value}"]
        args += [self.container, "bash", "-c", command]
        process = await asyncio.create_subprocess_exec(
            "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
        except TimeoutError:
            process.kill()
            raise
        return ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode or 0,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await _docker("cp", str(source_path), f"{self.container}:{target_path}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self.exec(f"mkdir -p {shlex.quote(target_dir)}", user="root")
        await _docker("cp", f"{source_dir}/.", f"{self.container}:{target_dir}")


async def _image_id(image: str) -> str:
    return (await _docker("image", "inspect", "--format", "{{.Id}}", image)).strip()


async def _image_exists(image: str) -> bool:
    process = await asyncio.create_subprocess_exec(
        "docker", "image", "inspect", image, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    return await process.wait() == 0


async def _docker(*args: str, check: bool = True) -> str:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if check and process.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed ({process.returncode}): {stderr.decode(errors='replace').strip()}")
    return stdout.decode(errors="replace")
