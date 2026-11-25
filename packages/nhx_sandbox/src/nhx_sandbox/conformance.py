# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conformance suite, which checks a live backend against what the ``SandboxBackend`` docstrings promise.

It runs against the local backend in CI, and each real backend's tests run it against that backend,
so a new backend, or a new release of an existing one, is checked the same way. A case that doesn't
apply to a backend is skipped based on its ``Capabilities``, never on which backend it is.

Cases share one sandbox where they can, because creating sandboxes is what makes the suite slow on a
real cluster. Each case works in its own scratch directory inside the sandbox's default working
directory, which every backend must make writable.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import tarfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from nhx_sandbox.core.backend import (
    EgressControl,
    ExecRequest,
    ExecResult,
    PlatformEgress,
    SandboxBackend,
    SandboxPhase,
    SandboxRef,
    SandboxSpec,
    WorkloadNeeds,
    WorkloadUser,
)
from nhx_sandbox.core.errors import RevisionConflict, SandboxLost, SandboxNotFound, UnsupportedOperation
from nhx_sandbox.egress import EgressPolicy, EgressRule

__all__ = ["CaseResult", "EgressProbe", "run_suite"]

Outcome = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True)
class EgressProbe:
    """How to check that a backend enforces egress. Without one, the suite skips those checks.

    Attributes:
        allowed_host: A host the sandbox can reach once it's allowed, such as ``pypi.org``.
        denied_host: A host the sandbox must not reach while it isn't listed.
        internal_targets: ``host:port`` addresses no sandbox should reach under a deny policy: the
            cloud metadata service and the Kubernetes API by default, the usual first stops for
            code trying to get out of a sandbox.
        fetch: A shell command that exits 0 only if ``{url}`` answered. It must run on the suite's
            image; the default uses Python, which the default image has.
        connect: A shell command that exits 0 only if a TCP connection to ``{target}`` opened. It
            connects directly, so a backend that silently redirects every connection to a proxy
            needs its own.
    """

    allowed_host: str
    denied_host: str
    internal_targets: tuple[str, ...] = ("169.254.169.254:80", "kubernetes.default.svc:443")
    fetch: str = 'python3 -c "import sys, urllib.request as u; u.urlopen(sys.argv[1], timeout=10)" {url}'
    connect: str = (
        "python3 -c \"import socket, sys; host, _, port = sys.argv[1].rpartition(':'); "
        'socket.create_connection((host, int(port)), timeout=5)" {target}'
    )


@dataclass(frozen=True)
class CaseResult:
    """One case's outcome, and what it saw if it didn't pass."""

    name: str
    outcome: Outcome
    detail: str = ""
    seconds: float = 0.0


class _Skip(Exception):
    """Raised by a case that doesn't apply to the backend being tested."""


@dataclass
class _Context:
    backend: SandboxBackend
    ref: SandboxRef
    spec: SandboxSpec
    scratch: str
    egress: EgressProbe | None
    extra_refs: list[SandboxRef] = field(default_factory=list)

    async def sh(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        user: str | int | None = None,
        stdin: bytes | None = None,
    ) -> ExecResult:
        request = ExecRequest(command=command, cwd=cwd, env=env or {}, timeout_s=timeout_s, user=user, stdin=stdin)
        return await self.backend.exec(self.ref, request)

    def path(self, name: str) -> str:
        return f"{self.scratch}/{name}"


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _exec_basics(ctx: _Context) -> None:
    result = await ctx.sh("echo out; echo err >&2; exit 7")
    _expect(result.exit_code == 7, f"exit code {result.exit_code}, wanted 7")
    _expect(result.stdout.strip() == "out", f"stdout {result.stdout!r}")
    _expect(result.stderr.strip() == "err", f"stderr {result.stderr!r}")
    _expect(not result.timed_out, "reported a timeout")


async def _exec_env_and_cwd(ctx: _Context) -> None:
    result = await ctx.sh(
        'printf "%s|%s|%s" "$CONFORMANCE_SPEC" "$CONFORMANCE_CALL" "$(pwd)"',
        cwd=ctx.scratch,
        env={"CONFORMANCE_CALL": "call"},
    )
    _expect(result.stdout == f"spec|call|{ctx.scratch}", f"got {result.stdout!r}")
    overridden = await ctx.sh('printf %s "$CONFORMANCE_SPEC"', env={"CONFORMANCE_SPEC": "call wins"})
    _expect(overridden.stdout == "call wins", f"call env didn't override spec env: {overridden.stdout!r}")


async def _exec_stdin(ctx: _Context) -> None:
    payload = b"line one\nline two\n"
    result = await ctx.sh("wc -l", stdin=payload)
    _expect(result.stdout.strip() == "2", f"wc -l saw {result.stdout!r}")


async def _exec_timeout_kills(ctx: _Context) -> None:
    """A timeout kills the command and anything it started in the background, not just the shell."""
    late, child = ctx.path("late"), ctx.path("child")
    started = time.monotonic()
    result = await ctx.sh(f"(sleep 3; touch {child}) & sleep 4; touch {late}", timeout_s=1)
    elapsed = time.monotonic() - started
    _expect(result.timed_out, f"not reported as timed out (exit {result.exit_code})")
    _expect(elapsed < 30, f"took {elapsed:.0f}s to return")
    await asyncio.sleep(6)
    survivors = await ctx.sh(f"for f in {late} {child}; do [ -e $f ] && echo $f; done; true")
    _expect(
        not survivors.stdout.strip(), f"processes kept running after the timeout and wrote {survivors.stdout.split()}"
    )
    after = await ctx.sh("echo ok")
    _expect(after.stdout.strip() == "ok", "the sandbox wasn't usable after a command timed out")


async def _exec_cancel_stops_command(ctx: _Context) -> None:
    """Cancelling ``exec`` stops the command, so it can't change the sandbox after the caller gave up on it.

    A backend that can't stop it may instead mark the sandbox unusable, so this case runs after every
    other case that uses the suite's sandbox.
    """
    marker = ctx.path("cancelled")
    call = asyncio.create_task(ctx.sh(f"sleep 4; touch {marker}"))
    await asyncio.sleep(1)
    call.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await call
    await asyncio.sleep(6)
    try:
        check = await ctx.sh(f"test -e {marker}")
    except SandboxLost:
        return
    _expect(check.exit_code != 0, "the command kept running after its call was cancelled")


async def _file_round_trip(ctx: _Context) -> None:
    data = bytes(range(256)) * 64
    path = ctx.path("nested/dir/blob.bin")
    await ctx.backend.write_file(ctx.ref, path, data, mode=0o750)
    _expect(await ctx.backend.read_file(ctx.ref, path) == data, "bytes changed in the round trip")
    # find -perm with a bare mode matches it exactly, on GNU and BSD alike; stat's flags differ between them.
    mode = await ctx.sh(f"find {path} -perm 750")
    _expect(mode.stdout.strip() == path, f"mode isn't 750: {(await ctx.sh(f'ls -l {path}')).stdout.strip()}")


async def _large_file_round_trip(ctx: _Context) -> None:
    """A file several times a gateway's message cap (OpenShell's is 1 MiB) must still arrive whole."""
    data = hashlib.sha256(b"nhx").digest() * (3 * 1024 * 1024 // 32 + 7)
    path = ctx.path("large.bin")
    await ctx.backend.write_file(ctx.ref, path, data)
    back = await ctx.backend.read_file(ctx.ref, path)
    _expect(len(back) == len(data) and back == data, f"wrote {len(data)} bytes, read back {len(back)}")
    size = await ctx.sh(f"wc -c < {path}")
    _expect(size.stdout.strip() == str(len(data)), f"the sandbox holds {size.stdout.strip()} bytes")


def _sample_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:

        def add(name: str, kind: bytes, *, data: bytes = b"", mode: int = 0o644, target: str = "") -> None:
            info = tarfile.TarInfo(name)
            info.type, info.mode = kind, mode
            if kind == tarfile.SYMTYPE:
                info.linkname = target
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data) if data else None)

        add("bin", tarfile.DIRTYPE, mode=0o755)
        add("bin/run.sh", tarfile.REGTYPE, data=b"#!/bin/sh\necho ok\n", mode=0o755)
        add("empty", tarfile.DIRTYPE, mode=0o755)
        add("data.txt", tarfile.REGTYPE, data=b"hello\n")
        add("link", tarfile.SYMTYPE, target="bin/run.sh", mode=0o777)
    return buffer.getvalue()


def _members(archive: bytes) -> dict[str, tuple[str, int, bytes | str]]:
    found: dict[str, tuple[str, int, bytes | str]] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.removeprefix("./")
            if name in ("", "."):
                continue
            if member.issym():
                found[name] = ("link", 0, member.linkname)
            elif member.isdir():
                found[name] = ("dir", member.mode & 0o777, b"")
            else:
                handle = tar.extractfile(member)
                found[name] = ("file", member.mode & 0o777, handle.read() if handle else b"")
    return found


async def _dir_round_trip(ctx: _Context) -> None:
    target = ctx.path("tree")
    archive = _sample_archive()
    await ctx.backend.upload_dir(ctx.ref, target, archive)
    run = await ctx.sh(f"{target}/link")
    _expect(run.stdout.strip() == "ok", f"the uploaded script didn't run: {run.stdout!r} {run.stderr!r}")
    sent, got = _members(archive), _members(await ctx.backend.download_dir(ctx.ref, target))
    _expect(sent == got, f"tree changed in the round trip: sent {sorted(sent)} got {sorted(got)}")


async def _refuses_unknown_users(ctx: _Context) -> None:
    caps = ctx.backend.capabilities
    if caps.exec_as_root and caps.exec_as_other_users:
        raise _Skip("the backend can run commands as any user")
    for user, allowed in ((0, caps.exec_as_root), (54321, caps.exec_as_other_users)):
        if allowed:
            continue
        try:
            await ctx.sh("id -u", user=user)
        except UnsupportedOperation:
            continue
        raise AssertionError(f"running as {user} wasn't refused")


async def _workload_user(ctx: _Context) -> None:
    caps = ctx.backend.capabilities
    if caps.workload_user is WorkloadUser.IMAGE:
        raise _Skip("the image's USER decides, so there is nothing backend-wide to check")
    uid = (await ctx.sh("id -u")).stdout.strip()
    if caps.workload_user is WorkloadUser.FIXED:
        _expect(uid == str(caps.fixed_uid), f"ran as uid {uid}, wanted {caps.fixed_uid}")
    else:
        _expect(uid.isdigit() and uid != "0", f"ran as uid {uid!r}, wanted an assigned non-root uid")


async def _check_and_refusals(ctx: _Context) -> None:
    """``check`` accepts the suite's own spec, reports what the backend can't do, and ``create`` refuses it."""
    caps = ctx.backend.capabilities
    fits = await ctx.backend.check(ctx.spec)
    _expect(fits == [], f"check refused the spec the suite runs on: {fits}")
    if not caps.exec_as_root:
        root = await ctx.backend.check(ctx.spec, WorkloadNeeds(runs_as_root=True))
        _expect(bool(root), "check accepted a task that runs as root on a backend that never runs as root")
    # Whichever kind of egress the backend can't honour: a per-sandbox backend can't take the platform's
    # rules on trust, and a backend without per-sandbox egress can't apply a policy.
    egress = PlatformEgress(reason="conformance") if caps.egress_per_sandbox else EgressPolicy()
    mismatched = replace(ctx.spec, name_hint="conf-refuse", egress=egress)
    _expect(bool(await ctx.backend.check(mismatched)), f"check accepted {type(egress).__name__} egress")
    try:
        ref = await ctx.backend.create(mismatched)
    except UnsupportedOperation:
        return
    ctx.extra_refs.append(ref)
    raise AssertionError(f"create accepted {type(egress).__name__} egress that check refused")


async def _get_and_list(ctx: _Context) -> None:
    info = await ctx.backend.get(ctx.ref)
    _expect(info.ref.id == ctx.ref.id, "get returned another sandbox")
    run_label = dict(ctx.spec.labels)
    listed = await ctx.backend.list_sandboxes(run_label)
    _expect(any(item.ref.id == ctx.ref.id for item in listed), f"list({run_label}) didn't include the sandbox")
    other = await ctx.backend.list_sandboxes({**run_label, "nhx-conformance-absent": "1"})
    _expect(not any(item.ref.id == ctx.ref.id for item in other), "list matched on a label the sandbox lacks")


async def _egress_denied_by_default(ctx: _Context) -> None:
    """The suite's own sandbox reaches nothing its egress policy doesn't list, internal addresses included."""
    if ctx.egress is None:
        raise _Skip("no egress probe was given")
    if not isinstance(ctx.spec.egress, EgressPolicy):
        raise _Skip("the sandbox runs under the platform's network rules, which the suite can't know")
    listed = {rule.target for rule in ctx.spec.egress.rules}
    if ctx.egress.denied_host in listed:
        raise _Skip(f"the spec's egress policy allows {ctx.egress.denied_host}")
    denied = await ctx.sh(ctx.egress.fetch.format(url=f"https://{ctx.egress.denied_host}/"), timeout_s=60)
    _expect(denied.exit_code != 0, f"{ctx.egress.denied_host} was reachable though the policy doesn't list it")
    for target in ctx.egress.internal_targets:
        reached = await ctx.sh(ctx.egress.connect.format(target=target), timeout_s=60)
        _expect(reached.exit_code != 0, f"the sandbox opened a connection to {target}")


async def _egress_live_update(ctx: _Context) -> None:
    if not isinstance(ctx.backend, EgressControl) or not ctx.backend.capabilities.egress_live_update:
        raise _Skip("the backend can't change egress while a sandbox runs")
    control: EgressControl = ctx.backend
    before = await control.get_egress(ctx.ref)
    allow = (
        EgressPolicy(rules=(EgressRule(target=ctx.egress.allowed_host),))
        if ctx.egress
        else EgressPolicy(rules=(EgressRule(target="example.org"),))
    )
    applied = await control.set_egress(ctx.ref, allow, expected_revision=before.revision)
    _expect(applied.revision != before.revision, "the revision didn't change")
    try:
        await control.set_egress(ctx.ref, EgressPolicy(), expected_revision=before.revision)
    except RevisionConflict:
        pass
    else:
        raise AssertionError("a stale expected_revision was accepted")
    if ctx.egress is None:
        return
    fetch = ctx.egress.fetch
    allowed = await ctx.sh(fetch.format(url=f"https://{ctx.egress.allowed_host}/"), timeout_s=60)
    _expect(allowed.exit_code == 0, f"allowed host unreachable: {allowed.stderr[-300:]}")
    denied = await ctx.sh(fetch.format(url=f"https://{ctx.egress.denied_host}/"), timeout_s=60)
    _expect(denied.exit_code != 0, "an unlisted host was reachable")
    current = await control.get_egress(ctx.ref)
    await control.set_egress(ctx.ref, EgressPolicy(), expected_revision=current.revision)
    after = await ctx.sh(fetch.format(url=f"https://{ctx.egress.allowed_host}/"), timeout_s=60)
    _expect(after.exit_code != 0, "a revoked host was still reachable")


async def _delete_is_idempotent(ctx: _Context) -> None:
    spec = replace(ctx.spec, name_hint="conf-del", labels={**ctx.spec.labels, "nhx-conformance-case": "delete"})
    ref = await ctx.backend.create(spec)
    ctx.extra_refs.append(ref)
    await ctx.backend.delete(ref)
    await ctx.backend.delete(ref)
    try:
        info = await ctx.backend.get(ref)
    except SandboxNotFound:
        return
    _expect(info.phase is SandboxPhase.STOPPED, f"a deleted sandbox reports {info.phase.value}, not stopped")


Case = Callable[[_Context], Awaitable[None]]

CASES: tuple[tuple[str, Case], ...] = (
    ("exec_basics", _exec_basics),
    ("exec_env_and_cwd", _exec_env_and_cwd),
    ("exec_stdin", _exec_stdin),
    ("exec_timeout_kills", _exec_timeout_kills),
    ("file_round_trip", _file_round_trip),
    ("large_file_round_trip", _large_file_round_trip),
    ("dir_round_trip", _dir_round_trip),
    ("refuses_unknown_users", _refuses_unknown_users),
    ("workload_user", _workload_user),
    ("check_and_refusals", _check_and_refusals),
    ("get_and_list", _get_and_list),
    ("egress_denied_by_default", _egress_denied_by_default),
    ("egress_live_update", _egress_live_update),
    ("exec_cancel_stops_command", _exec_cancel_stops_command),
    ("delete_is_idempotent", _delete_is_idempotent),
)


async def run_suite(
    backend: SandboxBackend,
    spec: SandboxSpec,
    *,
    egress: EgressProbe | None = None,
    only: frozenset[str] | None = None,
) -> list[CaseResult]:
    """Run every case against ``backend`` and return the outcomes. A failed case is a result, not an exception.

    Args:
        backend: The backend to test.
        spec: The sandbox to create. Its labels must be unique to this run, because ``get_and_list``
            lists by them. ``CONFORMANCE_SPEC=spec`` is added to its env.
        egress: How to check that egress is enforced. With ``None``, the suite only checks that egress
            can be changed and read back.
        only: Run just these cases.
    """
    spec = replace(spec, env={**spec.env, "CONFORMANCE_SPEC": "spec"})
    ref = await backend.create(spec)
    results: list[CaseResult] = []
    ctx: _Context | None = None
    try:
        home = await backend.exec(ref, ExecRequest(command="pwd"))
        scratch = f"{home.stdout.strip().rstrip('/')}/conformance-{uuid.uuid4().hex[:8]}"
        await backend.exec(ref, ExecRequest(command=f"mkdir -p {scratch}"))
        ctx = _Context(backend=backend, ref=ref, spec=spec, scratch=scratch, egress=egress)
        for name, case in CASES:
            if only is not None and name not in only:
                continue
            started = time.monotonic()
            try:
                await case(ctx)
                outcome, detail = "passed", ""
            except _Skip as skip:
                outcome, detail = "skipped", str(skip)
            except Exception as exc:  # noqa: BLE001 - a failed case is a result, not a crash
                outcome, detail = "failed", f"{type(exc).__name__}: {exc}"
            results.append(CaseResult(name, outcome, detail, round(time.monotonic() - started, 1)))
    finally:
        for extra in ctx.extra_refs if ctx else ():
            await backend.delete(extra)
        await backend.delete(ref)
    return results
