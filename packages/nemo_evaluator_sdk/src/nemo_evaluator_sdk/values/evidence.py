# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence value types shared by protocol metrics and agent evaluations."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import os
import shutil
import signal
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias, overload
from urllib.parse import urlparse

from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, model_validator

from nemo_evaluator_sdk.values.atif import FinalMetrics, Step, ToolCall, Trajectory
from nemo_evaluator_sdk.values.otlp import (
    parse_resource_spans,
    resource_spans_from_request,
    resource_spans_from_text,
    validate_resource_spans,
)

# Standard evidence keys shared by inference, persistence, and agent evaluation.
EVIDENCE_INITIAL_STATE = "initial_state"
EVIDENCE_TRACE = "trace"
EVIDENCE_LOGS = "logs"
EVIDENCE_FINAL_STATE = "final_state"
EVIDENCE_VERIFIER_LOGS = "verifier_logs"
EVIDENCE_RAW_STREAM = "raw_stream"
EVIDENCE_STREAM_EVENTS = "stream_events"
EVIDENCE_REQUEST_PAYLOAD = "request_payload"
EVIDENCE_REQUEST_HEADERS = "request_headers"
EVIDENCE_HTTP_METADATA = "http_metadata"
EVIDENCE_TRANSLATION_ERROR = "translation_error"

EVIDENCE_FORMAT_ATIF = "atif"
EVIDENCE_FORMAT_JSON = "json"
EVIDENCE_FORMAT_OTLP = "otlp"
EVIDENCE_FORMAT_TEXT = "text"

# Well-known evidence keys used by the core agent-eval artifact contract.
WellKnownEvidenceKey = Literal["initial_state", "trace", "logs", "final_state", "verifier_logs"]


class FilesystemEntry(BaseModel):
    """One path that differs between two filesystem snapshots."""

    model_config = ConfigDict(extra="forbid")

    path: str
    change_type: Literal["added", "modified", "deleted"]


class FilesystemDiff(BaseModel):
    """Set of paths that changed between two filesystem snapshots."""

    model_config = ConfigDict(extra="forbid")

    entries: list[FilesystemEntry] = Field(default_factory=list)

    def changed(
        self,
        *,
        prefix: str | None = None,
        kinds: set[str] | None = None,
    ) -> list[FilesystemEntry]:
        """Return entries optionally filtered by path prefix and change kind."""
        return [
            entry
            for entry in self.entries
            if (prefix is None or entry.path.startswith(prefix)) and (kinds is None or entry.change_type in kinds)
        ]


class CommandResult(BaseModel):
    """Outcome of running a verifier command against filesystem evidence."""

    model_config = ConfigDict(extra="forbid")

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Whether the command exited 0 without timing out."""
        return self.exit_code == 0 and not self.timed_out


class LocalFilesystemEvidence:
    """Constrained local filesystem handle for metric evidence access."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        """Resolved root path for this local evidence handle."""
        return self._root

    def path(self, relative_path: str | Path = ".") -> Path:
        """Return a path under the evidence root, rejecting traversal outside it."""
        relative = Path(relative_path)
        candidate = relative.resolve() if relative.is_absolute() else (self._root / relative).resolve()
        if not self._within_root(candidate):
            raise ValueError(f"evidence path {relative_path!r} resolves outside evidence root")
        return candidate

    def _within_root(self, path: Path) -> bool:
        """Whether ``path`` (after resolving symlinks) stays inside the evidence root."""
        resolved = path.resolve()
        return resolved == self._root or self._root in resolved.parents

    async def exists(self, relative_path: str | Path = ".") -> bool:
        """Return whether a path exists under the evidence root."""
        path = self.path(relative_path)
        return await asyncio.to_thread(path.exists)

    async def read_text(self, relative_path: str | Path, *, encoding: str = "utf-8") -> str:
        """Read a text file under the evidence root."""
        path = self.path(relative_path)
        return await asyncio.to_thread(path.read_text, encoding=encoding)

    async def iter_paths(self, relative_path: str | Path = ".", *, recursive: bool = False) -> list[str]:
        """List entries (files and directories) rooted at ``relative_path``."""
        base = self.path(relative_path)
        return await asyncio.to_thread(self._iter_paths_sync, base, recursive)

    def _iter_paths_sync(self, base: Path, recursive: bool) -> list[str]:
        if base.is_file():
            return [base.relative_to(self._root).as_posix()]
        iterator = base.rglob("*") if recursive else base.iterdir()
        return sorted(path.relative_to(self._root).as_posix() for path in iterator)

    async def read_bytes(self, relative_path: str | Path) -> bytes:
        """Read a binary file under the evidence root."""
        path = self.path(relative_path)
        return await asyncio.to_thread(path.read_bytes)

    async def list_files(self, pattern: str = "**/*") -> list[str]:
        """List relative posix paths of files (not directories) matching ``pattern``."""
        return await asyncio.to_thread(self._list_sync, pattern)

    def _list_sync(self, pattern: str) -> list[str]:
        return sorted(
            path.relative_to(self._root).as_posix()
            for path in self._root.glob(pattern)
            if path.is_file() and self._within_root(path)
        )

    async def diff(self, other: LocalFilesystemEvidence) -> FilesystemDiff:
        """Diff this snapshot (before) against ``other`` (after) by file content hash."""
        return await asyncio.to_thread(self._diff_sync, other)

    def _diff_sync(self, other: LocalFilesystemEvidence) -> FilesystemDiff:
        before = self._hashes()
        after = other._hashes()
        entries = [FilesystemEntry(path=path, change_type="added") for path in sorted(after.keys() - before.keys())]
        entries += [FilesystemEntry(path=path, change_type="deleted") for path in sorted(before.keys() - after.keys())]
        entries += [
            FilesystemEntry(path=path, change_type="modified")
            for path in sorted(before.keys() & after.keys())
            if before[path] != after[path]
        ]
        return FilesystemDiff(entries=entries)

    async def unified_diff(
        self,
        other: LocalFilesystemEvidence,
        relative_path: str | Path,
        *,
        context: int = 3,
    ) -> str:
        """Unified diff of one path between this snapshot (before) and ``other`` (after); ``""`` if identical or binary."""
        return await asyncio.to_thread(self._unified_diff_sync, other, relative_path, context)

    def _unified_diff_sync(self, other: LocalFilesystemEvidence, relative_path: str | Path, context: int) -> str:
        before_path, after_path = self.path(relative_path), other.path(relative_path)
        before = before_path.read_bytes() if before_path.is_file() else b""
        after = after_path.read_bytes() if after_path.is_file() else b""
        if before == after:
            return ""
        try:
            before_lines = before.decode("utf-8").splitlines(keepends=True)
            after_lines = after.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            return ""  # binary content: no textual patch
        rel = Path(relative_path).as_posix()
        return "".join(
            difflib.unified_diff(before_lines, after_lines, fromfile=f"a/{rel}", tofile=f"b/{rel}", n=context)
        )

    def _hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for path in self._safe_files():
            hashes[path.relative_to(self._root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes

    def _safe_files(self) -> list[Path]:
        """Regular files under the root, skipping symlinks whose target escapes it."""
        files: list[Path] = []
        for dirpath, _dirnames, filenames in os.walk(self._root, followlinks=False):
            for name in filenames:
                full = Path(dirpath) / name
                if full.is_symlink() and not self._within_root(full):
                    continue
                files.append(full)
        return files

    async def run_verifier(
        self,
        command: list[str],
        *,
        cwd: str = ".",
        overlay_files: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> CommandResult:
        """Run ``command`` (no shell) against a throwaway copy of the evidence; not a sandbox (host privileges).

        ``overlay_files`` is a ``{relative_path: contents}`` map of trusted files written *over* the copy
        after it is made, before the command runs. This is how a grader supplies held-out artifacts (a
        canonical test suite, a reference implementation) that must not live in — and cannot be edited
        through — the agent's own workspace. Overlay paths that escape the copy are rejected.
        """
        sandbox = Path(tempfile.mkdtemp(prefix="evidence-verify-")).resolve()
        try:
            workdir = (sandbox / cwd).resolve()
            if workdir != sandbox and sandbox not in workdir.parents:
                raise ValueError(f"verifier cwd {cwd!r} resolves outside evidence overlay")
            # symlinks=True copies links as-is (no host deref); the ignore hook drops links whose
            # target escapes the evidence root so the verifier can't read or write through them.
            await asyncio.to_thread(
                shutil.copytree,
                self._root,
                sandbox,
                dirs_exist_ok=True,
                symlinks=True,
                ignore=self._ignore_escaping_symlinks,
            )
            if overlay_files:
                await asyncio.to_thread(_write_overlay_files, sandbox, overlay_files)
            return await self._exec(command, workdir, timeout_s)
        finally:
            await asyncio.to_thread(shutil.rmtree, sandbox, True)

    def _ignore_escaping_symlinks(self, directory: str, names: list[str]) -> set[str]:
        """copytree ignore hook: drop absolute symlinks and links whose target escapes the evidence root."""
        ignored: set[str] = set()
        for name in names:
            full = Path(directory) / name
            if not full.is_symlink():
                continue
            if os.path.isabs(os.readlink(full)) or not self._within_root(full):
                ignored.add(name)
        return ignored

    @staticmethod
    async def _exec(command: list[str], cwd: Path, timeout_s: float | None) -> CommandResult:
        # start_new_session: child leads its own process group so a timeout can reap the whole tree.
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except TimeoutError:
            # wait_for leaves the tree running; kill the whole process group.
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            return CommandResult(exit_code=-1, timed_out=True)
        return CommandResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )


def _write_overlay_files(root: Path, files: Mapping[str, str]) -> None:
    """Write ``{relative_path: contents}`` into ``root``, rejecting paths that escape it."""
    resolved_root = root.resolve()
    for rel_path, contents in files.items():
        target = (resolved_root / str(rel_path)).resolve()
        if target != resolved_root and resolved_root not in target.parents:
            raise ValueError(f"overlay file path escapes the verifier copy: {rel_path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")


class EvidenceDescriptor(BaseModel):
    """Descriptor for a candidate trace, source, or artifact."""

    # ``anyOf`` mirrors the ``_requires_ref_or_data`` validator into the OpenAPI schema, so a payload
    # with neither ``ref`` nor ``data`` is rejected by the contract, not just at runtime.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"anyOf": [{"required": ["ref"]}, {"required": ["data"]}]},
    )

    kind: str = Field(description="Evidence type, e.g. 'filesystem', 'trace', 'log_bundle', or 'review'.")
    ref: str | None = Field(
        default=None,
        description="Reference to externally stored evidence (e.g. a local path or storage ref).",
    )
    format: str | None = Field(
        default=None,
        description="Parser hint for the evidence payload, e.g. 'atif' for normalized traces.",
    )
    data: JsonValue | None = Field(
        default=None,
        description="Small inline evidence payload; at least one of ref or data must be set.",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of this evidence artifact.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata associated with the evidence descriptor.",
    )

    @model_validator(mode="after")
    def _requires_ref_or_data(self) -> EvidenceDescriptor:
        if self.ref is None and self.data is None:
            raise ValueError("evidence descriptor requires ref or data")
        return self


# ATIF ingest: producers emit conformant ATIF (see values/atif.py, RFC 0001); we validate on read, not normalize.
def parse_atif(payload: Any) -> Trajectory:
    """Validate a payload as a canonical ATIF :class:`Trajectory` (raises ``ValidationError`` if non-conformant)."""
    return payload if isinstance(payload, Trajectory) else Trajectory.model_validate(payload)


logger = logging.getLogger(__name__)


def read_atif(path: Path) -> Trajectory | None:
    """The ATIF trajectory at ``path``, or ``None`` if it is absent, unreadable, or not ATIF.

    A producer whose trajectory filename does not identify the format has to read the file to know
    what it holds: that a file exists says nothing about whether it is ATIF.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return parse_atif(payload)
    except ValueError:
        logger.warning("Trajectory at %s is not valid ATIF; ignoring it.", path)
        return None


def read_otlp_spans(path: Path) -> list[ResourceSpans] | None:
    """The OTLP trace at ``path``, or ``None`` if it is absent, unreadable, or not OTLP.

    The tolerant counterpart to :class:`OTLPTraceHandle`, for callers assembling a trial rather than
    scoring one: a trace that will not parse costs them a field, not the trial.
    """
    try:
        return parse_resource_spans(resource_spans_from_text(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable OTLP trace %s: %s", path, error)
        return None


def final_agent_message(trajectory: Trajectory | None) -> str | None:
    """The agent's last message in an ATIF trajectory, which is its answer for the trial.

    Only the *last* agent step can be the answer. An earlier one is intermediate reasoning, so an
    agent that ends on an empty message has produced no answer rather than the previous one.
    """
    if trajectory is None:
        return None
    for step in reversed(trajectory.steps):
        if step.source == "agent":
            return step.message or None
    return None


class OTLPTraceHandle:
    """Lazily loaded read handle over OTLP/JSON (JSONL or a single object)."""

    format: Literal["otlp"] = "otlp"

    def __init__(self, descriptor: EvidenceDescriptor) -> None:
        self._descriptor = descriptor
        self._resource_spans: list[ResourceSpans] | None = None

    async def resource_spans(self) -> list[ResourceSpans]:
        """Return the trace as typed OTLP resource spans, read and parsed once.

        This is the representation producers and consumers share. A caller sending them to OTLP
        ingest wraps them in an ``ExportTraceServiceRequest``, which is that RPC's envelope and
        nothing a reader needs.

        Raises:
            ValueError: The evidence is missing, unreadable, or not valid OTLP.
        """
        if self._resource_spans is None:
            self._resource_spans = await asyncio.to_thread(self._read)
        return self._resource_spans

    def _read(self) -> list[ResourceSpans]:
        """Load inline or local OTLP evidence and parse it.

        The decoded JSON is a step on the way in, never a representation this hands out: OTLP is
        a schema, and a reader that has honoured it cannot be handed a dict that has not.
        """
        descriptor = self._descriptor
        if descriptor.data is not None:
            payload = (
                validate_resource_spans(descriptor.data)
                if isinstance(descriptor.data, list)
                else resource_spans_from_request(descriptor.data)
            )
        elif descriptor.ref is None:
            raise ValueError("trace evidence descriptor requires ref or data")
        else:
            payload = resource_spans_from_text(_local_filesystem_ref(descriptor.ref).read_text(encoding="utf-8"))
        return parse_resource_spans(payload)


class ATIFTraceHandle:
    """Lazily validated read handle exposing a trace descriptor as an ATIF :class:`Trajectory`."""

    format: Literal["atif"] = "atif"

    def __init__(self, descriptor: EvidenceDescriptor) -> None:
        self._descriptor = descriptor
        self._trajectory: Trajectory | None = None

    async def trace(self) -> Trajectory:
        """Return the ATIF trajectory, reading and validating on first access."""
        if self._trajectory is None:
            payload = await asyncio.to_thread(self._load_payload)
            self._trajectory = parse_atif(payload)
        return self._trajectory

    def _load_payload(self) -> Any:
        descriptor = self._descriptor
        if descriptor.data is not None:
            return descriptor.data
        if descriptor.ref is None:
            raise ValueError("trace evidence descriptor requires ref or data")
        return json.loads(_local_filesystem_ref(descriptor.ref).read_text(encoding="utf-8"))

    async def steps(self) -> list[Step]:
        """Return the ATIF steps in order."""
        return (await self.trace()).steps

    async def tool_calls(self) -> list[ToolCall]:
        """Return all tool calls flattened across agent steps, in order."""
        calls: list[ToolCall] = []
        for step in await self.steps():
            calls.extend(step.tool_calls or [])
        return calls

    async def token_usage(self) -> FinalMetrics:
        """Return aggregate token usage (trajectory ``final_metrics``, else summed per step)."""
        trajectory = await self.trace()
        if trajectory.final_metrics is not None:
            return trajectory.final_metrics
        prompt = sum((step.metrics.prompt_tokens or 0) for step in trajectory.steps if step.metrics is not None)
        completion = sum((step.metrics.completion_tokens or 0) for step in trajectory.steps if step.metrics is not None)
        return FinalMetrics(total_prompt_tokens=prompt or None, total_completion_tokens=completion or None)


TraceHandle: TypeAlias = ATIFTraceHandle | OTLPTraceHandle


class LogHandle:
    """Read handle over a log-bundle directory."""

    def __init__(self, root: str | Path) -> None:
        self._fs = LocalFilesystemEvidence(root)

    async def list_files(self) -> list[str]:
        """Return relative paths of log files in the bundle."""
        return await self._fs.list_files("**/*")

    async def read_text(self, name: str) -> str:
        """Read one log file's full text."""
        return await self._fs.read_text(name)

    async def tail(self, name: str, lines: int = 50) -> str:
        """Return the last ``lines`` lines of a log file."""
        text = await self._fs.read_text(name)
        return "\n".join(text.splitlines()[-lines:])


class CandidateEvidence(BaseModel):
    """Named evidence descriptors attached to an AgentEvalTrial."""

    model_config = ConfigDict(extra="forbid")

    descriptors: dict[str, EvidenceDescriptor] = Field(
        default_factory=dict,
        description="Evidence descriptors keyed by name (e.g. 'final_state', 'trace', 'logs').",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata associated with the evidence collection.",
    )
    _filesystem_cache: dict[str, LocalFilesystemEvidence] = PrivateAttr(default_factory=dict)
    _trace_cache: dict[str, TraceHandle] = PrivateAttr(default_factory=dict)
    _log_cache: dict[str, LogHandle] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_descriptor_mapping(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict) and "descriptors" not in value and "metadata" not in value:
            return {"descriptors": value}
        return value

    def names(self, *, kind: str | None = None) -> list[str]:
        """Return evidence names, optionally filtered by descriptor kind."""
        if kind is None:
            return list(self.descriptors)
        return [name for name, descriptor in self.descriptors.items() if descriptor.kind == kind]

    def get(self, name: str) -> EvidenceDescriptor | None:
        """Return a descriptor by name without materializing evidence."""
        return self.descriptors.get(name)

    def require(self, name: str, *, kind: str | None = None) -> EvidenceDescriptor:
        """Return a descriptor by name, raising when it is missing or has the wrong kind."""
        descriptor = self.get(name)
        if descriptor is None:
            raise KeyError(f"missing evidence descriptor {name!r}")
        if kind is not None and descriptor.kind != kind:
            raise ValueError(f"evidence descriptor {name!r} has kind {descriptor.kind!r}, expected {kind!r}")
        return descriptor

    async def filesystem(self, name: str) -> LocalFilesystemEvidence:
        """Return a cached local filesystem handle for a named filesystem descriptor."""
        cached = self._filesystem_cache.get(name)
        if cached is not None:
            return cached

        descriptor = self.require(name, kind="filesystem")
        if descriptor.ref is None:
            raise ValueError(f"filesystem evidence descriptor {name!r} requires a local ref")

        root = _local_filesystem_ref(descriptor.ref)
        handle = LocalFilesystemEvidence(root)
        self._filesystem_cache[name] = handle
        return handle

    def _resolve_trace_key(self, name: str, evidence_format: str | None) -> str:
        """Return the descriptor key holding ``name`` in ``evidence_format``.

        A producer carrying several encodings of one trace registers each under
        ``"{name}:{format}"`` as well as naming one of them primary under ``name``.
        ``name`` is tried first so a caller asking for the encoding that is already
        primary shares one cached handle instead of parsing the descriptor twice.
        A candidate is accepted on its declared ``format``, never on its key suffix,
        so a mis-filed descriptor cannot falsify the caller's narrowed return type.
        """
        if evidence_format is None:
            return name
        for key in (name, f"{name}:{evidence_format}"):
            descriptor = self.get(key)
            if descriptor is not None and (descriptor.format or EVIDENCE_FORMAT_ATIF) == evidence_format:
                return key
        raise KeyError(f"missing evidence descriptor {name!r} in format {evidence_format!r}")

    @overload
    async def trace(self, name: str = ..., *, format: None = ...) -> TraceHandle: ...

    @overload
    async def trace(self, name: str = ..., *, format: Literal["atif"]) -> ATIFTraceHandle: ...

    @overload
    async def trace(self, name: str = ..., *, format: Literal["otlp"]) -> OTLPTraceHandle: ...

    @overload
    async def trace(self, name: str = ..., *, format: str) -> TraceHandle: ...

    async def trace(self, name: str = "trace", *, format: str | None = None) -> TraceHandle:
        """Return a cached trace handle for a named trace descriptor (read lazily on first access).

        Args:
            name: Evidence key naming the trace's role, such as ``"trace"``.
            format: Encoding to read the trace as. When omitted, the descriptor stored
                under ``name`` decides, giving the producer's primary view.

        Returns:
            A handle over the trace, typed to the requested format when one is given.

        Raises:
            KeyError: No descriptor holds ``name`` in the requested format.
            ValueError: The resolved descriptor declares a format with no handle.
        """
        key = self._resolve_trace_key(name, format)
        cached = self._trace_cache.get(key)
        if cached is not None:
            return cached
        descriptor = self.require(key, kind=EVIDENCE_TRACE)
        handle: TraceHandle
        if descriptor.format is None or descriptor.format == EVIDENCE_FORMAT_ATIF:
            handle = ATIFTraceHandle(descriptor)
        elif descriptor.format == EVIDENCE_FORMAT_OTLP:
            handle = OTLPTraceHandle(descriptor)
        else:
            raise ValueError(f"unknown trace evidence format {descriptor.format!r}")
        self._trace_cache[key] = handle
        return handle

    async def logs(self, name: str = "logs") -> LogHandle:
        """Return a cached log-bundle handle for a named logs descriptor."""
        cached = self._log_cache.get(name)
        if cached is not None:
            return cached
        descriptor = self.require(name, kind="logs")
        if descriptor.ref is None:
            raise ValueError(f"logs evidence descriptor {name!r} requires a local ref")
        handle = LogHandle(_local_filesystem_ref(descriptor.ref))
        self._log_cache[name] = handle
        return handle


def _local_filesystem_ref(ref: str) -> Path:
    """Resolve a local ref (POSIX path, ``file://`` URI, or Windows drive path) to a Path; reject network/cloud URIs."""
    parsed = urlparse(ref)
    # A single-letter scheme is a Windows drive letter (e.g. "C:\\dir"), not a URI scheme.
    if len(parsed.scheme) == 1 and parsed.scheme.isalpha():
        return Path(ref)
    if parsed.scheme in {"http", "https", "s3", "gs"}:
        raise ValueError("CandidateEvidence.filesystem only supports local filesystem refs")
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme:
        raise ValueError(f"CandidateEvidence.filesystem does not support {parsed.scheme!r} refs")
    return Path(ref)
