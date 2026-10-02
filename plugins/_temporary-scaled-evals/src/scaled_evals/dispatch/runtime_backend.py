# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generic runtime-backend contract — no concrete backend lives here.

A :class:`RuntimeBackend` is the seam between the control plane and whatever
actually runs an evaluation. The control plane builds a :class:`LaunchSpec`
from the evaluation row and calls ``launch`` / ``status`` / ``teardown``;
everything cluster- or vendor-specific stays on the far side of this
interface. That isolation is what makes dispatch unit-testable: tests inject a
fake backend and never touch a cluster.

Concrete backends live in their own modules (e.g.
:mod:`scaled_evals.dispatch.sandbox_k8s`) and are exposed to the control plane
through :mod:`scaled_evals.dispatch.registry`. The registry owns runtime-name
lookup plus the backend capabilities that the API/worker need outside the core
launch/status/teardown lifecycle, such as artifact roots and log locations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from scaled_evals.models.resource_usage import ResourceUsageSample
from scaled_evals.models.runtime import LaunchHandle, LaunchSpec, ResultSummary, RuntimeStatus


class IncompatibleTaskError(ValueError):
    """The task, as finalized, can't run on the selected runtime.

    A launch that raises it fails with this class name as its failure code, which is always
    classified as a task failure, whatever words the message contains.
    """


@runtime_checkable
class RuntimeBackend(Protocol):
    """Contract every dispatch backend implements.

    A Protocol (not an ABC) so tests can supply any duck-typed fake without
    inheriting. The lifecycle calls are start a run, observe it, tear it down;
    ``summarize`` reduces a finished run's framework-typed result to the generic
    :class:`ResultSummary` the worker persists — keeping result interpretation
    on the framework-aware side of this seam, not in the worker.
    """

    name: str

    def launch(self, spec: LaunchSpec) -> LaunchHandle: ...

    def status(self, handle: LaunchHandle) -> RuntimeStatus: ...

    def teardown(self, handle: LaunchHandle) -> None: ...

    def summarize(self, result: Mapping[str, Any]) -> ResultSummary: ...


Submitter = Callable[[LaunchSpec], LaunchHandle]
StatusReader = Callable[[LaunchHandle], RuntimeStatus]
Terminator = Callable[[LaunchHandle], None]
ResultSummarizer = Callable[[Mapping[str, Any]], ResultSummary]
ResourceSampler = Callable[[LaunchHandle], Sequence[ResourceUsageSample]]


class CallableRuntimeBackend:
    """Optional backend implementation assembled from provider-owned callables.

    Runtime plugins may use this adapter to avoid repeating lifecycle delegation,
    but they do not have to inherit from it. :class:`RuntimeBackend` remains a
    structural protocol and the registry continues to accept any conforming
    backend factory.
    """

    name: str

    def __init__(
        self,
        *,
        name: str,
        summarizer: ResultSummarizer,
        submitter: Submitter | None = None,
        status_reader: StatusReader | None = None,
        terminator: Terminator | None = None,
        resource_sampler: ResourceSampler | None = None,
        launch_unavailable: str | None = None,
        status_unavailable: str | None = None,
        teardown_unavailable: str | None = None,
    ) -> None:
        self.name = name
        self._submitter = submitter
        self._status_reader = status_reader
        self._terminator = terminator
        self._resource_sampler = resource_sampler
        self._summarizer = summarizer
        self._launch_unavailable = launch_unavailable or f"{name} live submission is not wired"
        self._status_unavailable = status_unavailable or f"{name} status reads are not wired"
        self._teardown_unavailable = teardown_unavailable or f"{name} teardown is not wired"

    def launch(self, spec: LaunchSpec) -> LaunchHandle:
        if self._submitter is None:
            raise NotImplementedError(self._launch_unavailable)
        return self._submitter(spec)

    def status(self, handle: LaunchHandle) -> RuntimeStatus:
        if self._status_reader is None:
            raise NotImplementedError(self._status_unavailable)
        return self._status_reader(handle)

    def teardown(self, handle: LaunchHandle) -> None:
        if self._terminator is None:
            raise NotImplementedError(self._teardown_unavailable)
        self._terminator(handle)

    def summarize(self, result: Mapping[str, Any]) -> ResultSummary:
        return self._summarizer(result)

    def sample_resources(self, handle: LaunchHandle) -> Sequence[ResourceUsageSample]:
        if self._resource_sampler is None:
            return ()
        return self._resource_sampler(handle)


def _no_dispatch_work_dir(_: str) -> Path | None:
    return None


def _no_extra_log_paths(_: str) -> Sequence[Path]:
    return ()


@dataclass(frozen=True)
class RuntimeBackendCapabilities:
    """Runtime-specific metadata used by the generic control plane.

    Keep backend behavior behind :class:`RuntimeBackend`; use capabilities only
    for information the worker/API need before or after the backend lifecycle:
    where artifacts land, where dispatch logs land, which runner container may
    still be streaming logs, whether archive generation should run, and whether
    a launched run can be torn down.
    """

    artifact_root: Callable[[str], Path]
    dispatch_work_dir: Callable[[str], Path | None] = _no_dispatch_work_dir
    dispatch_log_name: str | None = None
    runner_container_prefix: str | None = None
    extra_dispatch_log_names: tuple[str, ...] = ()
    extra_log_paths: Callable[[str], Sequence[Path]] = _no_extra_log_paths
    supports_archive: bool = True
    supports_teardown: bool = True
    supported_network_policies: tuple[str, ...] = ("unrestricted",)

    def dispatch_log_path(self, evaluation_id: str) -> Path | None:
        if not self.dispatch_log_name:
            return None
        work_dir = self.dispatch_work_dir(evaluation_id)
        if work_dir is None:
            return None
        return work_dir / self.dispatch_log_name

    def runner_container_name(self, evaluation_id: str) -> str | None:
        if not self.runner_container_prefix:
            return None
        return f"{self.runner_container_prefix}-{evaluation_id}"

    def log_file_candidates(self, evaluation_id: str) -> list[Path]:
        """Return runtime-owned log files, excluding handle-provided paths."""
        paths: list[Path] = []
        if dispatch_log := self.dispatch_log_path(evaluation_id):
            paths.append(dispatch_log)

        artifact_root = self.artifact_root(evaluation_id)
        paths.extend(
            [
                artifact_root / "switchyard" / "switchyard.log",
                artifact_root / "switchyard" / "switchyard.previous.log",
                artifact_root / "switchyard" / "status.json",
            ]
        )

        if work_dir := self.dispatch_work_dir(evaluation_id):
            paths.extend(work_dir / name for name in self.extra_dispatch_log_names)

        paths.extend(self.extra_log_paths(evaluation_id))
        return paths


@dataclass(frozen=True)
class RuntimeBackendRegistration:
    """Registry entry for one runtime backend."""

    name: str
    factory: Callable[[], RuntimeBackend]
    capabilities: RuntimeBackendCapabilities
    validate: Callable[[], None] | None = None
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def build(self) -> RuntimeBackend:
        return self.factory()

    def validate_config(self) -> None:
        if self.validate is not None:
            self.validate()
            return
        # Backend factories already contain the settings validation needed to
        # build live submitters. Constructing a backend performs no network IO.
        self.factory()
