# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A ``SandboxBackend`` that runs each "sandbox" as a temporary directory on this machine.

It isolates nothing: commands run as this process's user, with this machine's network and
filesystem, and it reports ``Isolation.NONE``. It exists so the conformance suite and code built on
the protocol can be tested without a cluster. Never use it for untrusted code.

Egress is stored and read back but not enforced, so the conformance suite's egress checks are skipped.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import shutil
import signal
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from nhx_sandbox.core.archive import pack_dir, unpack_into
from nhx_sandbox.core.backend import (
    AppliedEgress,
    Capabilities,
    ExecRequest,
    ExecResult,
    Isolation,
    SandboxInfo,
    SandboxPhase,
    SandboxRef,
    SandboxSpec,
    WorkloadNeeds,
    WorkloadUser,
    capability_refusals,
    merged_env,
    require_absolute_path,
    require_egress_policy,
    require_own_ref,
)
from nhx_sandbox.core.errors import RevisionConflict, SandboxBackendError, SandboxNotFound, UnsupportedOperation
from nhx_sandbox.egress import EgressPolicy

NAME = "local"
# Variables kept from this process's environment so commands can find their tools. Nothing else leaks in.
_INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "TMPDIR")


@dataclass
class _Sandbox:
    ref: SandboxRef
    root: Path
    spec: SandboxSpec
    egress: EgressPolicy
    revision: int = 1
    labels: dict[str, str] = field(default_factory=dict)


def _kill_group(pgid: int) -> None:
    """Kill every process in the group, including any it forks while the kill is under way.

    ``killpg`` signals the group's processes one at a time. A shell whose child dies first can wake up
    and run its next command before its own signal arrives. Stopping the whole group first means
    nothing in it can fork while it's being killed.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGSTOP)
        os.killpg(pgid, signal.SIGKILL)


class LocalProcessBackend:
    """Sandboxes as temporary directories and commands as local processes. For tests only."""

    name = NAME
    capabilities = Capabilities(
        isolation=Isolation.NONE,
        exec_as_root=False,
        exec_as_other_users=False,
        workload_user=WorkloadUser.IMAGE,
        fixed_uid=None,
        server_ttl=False,
        egress_live_update=True,
        egress_readback=True,
        egress_tld_wildcards=True,
        gpus=False,
        multi_container=False,
        egress_per_sandbox=True,
        concurrent_exec=True,
    )

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir
        self._sandboxes: dict[str, _Sandbox] = {}
        self._revisions = itertools.count(2)

    def _require(self, ref: SandboxRef) -> _Sandbox:
        require_own_ref(ref, NAME)
        sandbox = self._sandboxes.get(ref.id)
        if sandbox is None:
            raise SandboxNotFound(f"sandbox {ref.name} is gone")
        return sandbox

    async def check(self, spec: SandboxSpec, needs: WorkloadNeeds = WorkloadNeeds()) -> list[str]:
        return capability_refusals(self.capabilities, spec, needs)

    async def create(self, spec: SandboxSpec) -> SandboxRef:
        if reasons := await self.check(spec):
            raise UnsupportedOperation("; ".join(reasons))
        egress = require_egress_policy(spec, NAME)
        root = Path(tempfile.mkdtemp(prefix=f"nhx-{spec.name_hint[:20]}-", dir=self._base_dir))
        sandbox_id = uuid.uuid4().hex
        ref = SandboxRef(backend=NAME, id=sandbox_id, name=root.name)
        self._sandboxes[sandbox_id] = _Sandbox(ref=ref, root=root, spec=spec, egress=egress, labels=dict(spec.labels))
        return ref

    async def get(self, ref: SandboxRef) -> SandboxInfo:
        sandbox = self._require(ref)
        return SandboxInfo(ref=ref, phase=SandboxPhase.READY, labels=dict(sandbox.labels))

    async def list_sandboxes(self, labels: Mapping[str, str]) -> list[SandboxInfo]:
        return [
            SandboxInfo(ref=sandbox.ref, phase=SandboxPhase.READY, labels=dict(sandbox.labels))
            for sandbox in self._sandboxes.values()
            if all(sandbox.labels.get(key) == value for key, value in labels.items())
        ]

    async def delete(self, ref: SandboxRef) -> None:
        require_own_ref(ref, NAME)
        sandbox = self._sandboxes.pop(ref.id, None)
        if sandbox is not None:
            await asyncio.to_thread(shutil.rmtree, sandbox.root, True)

    async def exec(self, ref: SandboxRef, request: ExecRequest) -> ExecResult:
        sandbox = self._require(ref)
        if request.user is not None:
            raise UnsupportedOperation("the local backend runs every command as this process's user")
        env = merged_env(
            {key: os.environ[key] for key in _INHERITED_ENV if key in os.environ},
            {"HOME": str(sandbox.root)},
            sandbox.spec.env,
            request.env,
        )
        cwd = request.cwd or sandbox.spec.workdir or str(sandbox.root)
        process = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            request.command,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE if request.stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # A process group of its own, so a timeout kills everything the command started.
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(request.stdin), timeout=request.timeout_s)
        except asyncio.CancelledError:
            # The caller gave up on the command, so it must not keep running and change the sandbox.
            _kill_group(process.pid)
            raise
        except TimeoutError:
            _kill_group(process.pid)
            stdout, stderr = await process.communicate()
            return ExecResult(
                exit_code=137,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                timed_out=True,
            )
        if process.returncode is None:
            raise SandboxBackendError("command ended without an exit code")
        return ExecResult(
            exit_code=process.returncode, stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace")
        )

    async def write_file(self, ref: SandboxRef, path: str, data: bytes, *, mode: int = 0o644) -> None:
        self._require(ref)
        require_absolute_path(path)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(mode)

    async def read_file(self, ref: SandboxRef, path: str) -> bytes:
        self._require(ref)
        require_absolute_path(path)
        try:
            return Path(path).read_bytes()
        except OSError as exc:
            raise SandboxBackendError(f"could not read {path}: {exc}") from exc

    async def upload_dir(self, ref: SandboxRef, path: str, archive: bytes) -> None:
        self._require(ref)
        require_absolute_path(path)
        await asyncio.to_thread(unpack_into, archive, Path(path))

    async def download_dir(self, ref: SandboxRef, path: str) -> bytes:
        self._require(ref)
        require_absolute_path(path)
        return await asyncio.to_thread(pack_dir, Path(path))

    async def get_egress(self, ref: SandboxRef) -> AppliedEgress:
        sandbox = self._require(ref)
        return AppliedEgress(revision=str(sandbox.revision), policy=sandbox.egress.model_dump(mode="json"))

    async def set_egress(
        self, ref: SandboxRef, policy: EgressPolicy, *, expected_revision: str | None = None
    ) -> AppliedEgress:
        sandbox = self._require(ref)
        if expected_revision is not None and expected_revision != str(sandbox.revision):
            raise RevisionConflict(f"egress changed since revision {expected_revision}; it's at {sandbox.revision} now")
        sandbox.egress = policy
        sandbox.revision = next(self._revisions)
        return await self.get_egress(ref)

    async def aclose(self) -> None:
        """Remove every directory this backend made; unlike a real sandbox, nothing else could ever find them."""
        for sandbox in list(self._sandboxes.values()):
            await self.delete(sandbox.ref)
