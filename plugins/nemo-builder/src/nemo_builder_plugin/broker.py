# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The credential broker's decisions: which step is asking, and what its job may have.

Everything here is pure -- no Kubernetes, no HTTP, no registry -- so the rules the broker enforces
are the rules the tests read. The process around them is ``run/broker.py``.

**Identity comes from the platform, not the step.** A step presents its pod's ServiceAccount
token; the API server says which pod it is bound to; the pod's labels, which Jobs wrote, say which
job, workspace and step it is. :func:`identify_pod` is the check on that pod.

**Entitlement is derived, never requested.** A step asks for credentials without naming a scope.
What it gets is a function of its role's :class:`~nemo_builder_plugin.backend.StepEntitlement` and
its job's ``pending`` rows, bounded by ``<repository_prefix>/<the job's workspace>/``. The bound
is checked even though the rows already name the destinations, because rows are data: anything
able to write a row must not be able to widen what the broker grants. The rows only ever narrow
it. A raw job has no rows, so it gets nothing.

For the same reason a row's repository and system tag are parsed, not trusted as strings. Both
end up in something the broker sends -- a token scope, a reference ``crane`` resolves -- where a
space would add a second scope and an ``@sha256:`` would name a digest of the writer's choosing.

ci-toolkit's broker is the counter-example this exists to avoid: it mints short-lived, one-repo
tokens for whatever repository the caller names, and nothing checks that the caller owns it. A
narrow token for a repository you are not entitled to is still a leak.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from nemo_builder_plugin.backend import StepEntitlement
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.identity import ImageIdentityError, split_system_tag, validate_repository, validate_tag

#: What Jobs writes on every step pod. Mirrors ``nhx.core.jobs`` constants; the broker runs without
#: the Jobs service installed, so it restates them rather than importing them.
MANAGED_BY_LABEL = "nhx.nvidia.com/managed_by"
MANAGED_BY_JOBS = "jobs-controller"
WORKSPACE_LABEL = "nhx.nvidia.com/job_workspace_id"
JOB_LABEL = "nhx.nvidia.com/job_id"
STEP_LABEL = "nhx.nvidia.com/job_step_name"
PROFILE_LABEL = "nhx.nvidia.com/job_execution_profile"

#: Pod phases after which no step is running. A bound token outlives the process -- it is valid
#: until the pod object is deleted, and Jobs keeps finished pods for a while -- so these refuse.
FINISHED_PHASES = frozenset({"Succeeded", "Failed", "Unknown"})


class NotIdentified(Exception):
    """The credential establishes no identity this broker serves.

    The message says which check failed -- never the credential -- because a refusal nobody can
    explain can't be audited.
    """


@dataclass(frozen=True, slots=True)
class Pod:
    """The facts about a pod the broker checks. All of them written by Jobs or the kubelet."""

    name: str
    uid: str
    service_account: str
    phase: str
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepIdentity:
    """A step's pod, as the platform created it: which job it belongs to, not what it claims."""

    workspace: str
    #: The job's name within its workspace.
    job: str
    pod: str
    #: The key of its entitlement in the backend's ``entitlements()``.
    role: str

    @property
    def job_ref(self) -> str:
        """``<workspace>/<job>``: what ``Provenance.job`` records."""
        return f"{self.workspace}/{self.job}"


def identify_pod(
    pod: Pod,
    *,
    token_uid: str,
    token_service_account: str,
    entitlements: Mapping[str, StepEntitlement],
) -> StepIdentity:
    """Which step ``pod`` is, or :class:`NotIdentified` naming the check that failed.

    ``token_uid`` and ``token_service_account`` are what the API server said about the token:
    the UID of the pod it is bound to, and the ServiceAccount it names.

    **Refuse finished pods; don't require ``Running``.** A step's first request comes
    milliseconds after its container starts, before the kubelet has reported the pod
    ``Running`` -- measured on minikube, where requiring ``Running`` refused every push. What must
    be refused is a pod that has *finished*, whose token stays valid until the pod is deleted.
    """
    if pod.uid != token_uid:
        raise NotIdentified(f"pod {pod.name} is not the pod the token was bound to: a new pod by that name")
    if pod.phase in FINISHED_PHASES:
        raise NotIdentified(f"pod {pod.name} has finished ({pod.phase})")
    labels = pod.labels
    if labels.get(MANAGED_BY_LABEL) != MANAGED_BY_JOBS:
        raise NotIdentified(
            f"pod {pod.name} was not created by Jobs ({MANAGED_BY_LABEL}={labels.get(MANAGED_BY_LABEL)!r})"
        )
    step, profile = labels.get(STEP_LABEL), labels.get(PROFILE_LABEL)
    role = next(
        (name for name, entitled in entitlements.items() if (entitled.step, entitled.profile) == (step, profile)), None
    )
    if role is None:
        raise NotIdentified(f"pod {pod.name}: step {step!r} under profile {profile!r} is entitled to nothing")
    expected = entitlements[role].service_account
    for what, found in (("pod", pod.service_account), ("token", token_service_account)):
        if found != expected:
            raise NotIdentified(f"pod {pod.name}: the {what} names ServiceAccount {found!r}, not {expected!r}")
    workspace, job = labels.get(WORKSPACE_LABEL), labels.get(JOB_LABEL)
    if not workspace or not job:
        raise NotIdentified(f"pod {pod.name} does not say which job and workspace it belongs to")
    return StepIdentity(workspace=workspace, job=job, pod=pod.name, role=role)


def _under(prefix: str, *parts: str) -> str:
    """``<prefix>/<parts...>/``, skipping empty parts, so an empty prefix bounds by workspace alone."""
    return "/".join(part for part in (prefix, *parts) if part) + "/"


def _within(repository: str, bound: str) -> bool:
    """``repository`` is a repository path, and under ``bound``."""
    try:
        validate_repository(repository)
    except ImageIdentityError:
        return False
    return repository.startswith(bound)


def _system_tag_refusal(tag: str, workspace: str) -> str | None:
    """Why ``tag`` is not a system tag of ``workspace``'s, or None if it is."""
    try:
        tag_workspace = split_system_tag(validate_tag(tag))[0]
    except ImageIdentityError:
        return f"system tag {tag!r} is not one this system composes"
    if tag_workspace != workspace:
        return f"system tag {tag!r} is not workspace {workspace}'s"
    return None


def entitled_rows(
    rows: Iterable[ContainerImage], identity: StepIdentity, *, registry: str, repository_prefix: str
) -> list[ContainerImage]:
    """The rows ``identity``'s job may publish: its own, ``pending``, on this registry, under its workspace.

    ``pending`` because a settled row has nothing left to publish: a credential for it could only
    overwrite what was verified. Rows of another job -- or registry, or workspace, or with a
    repository that is not a repository path -- are dropped, never an error, since whoever read
    them may have read more than this job's.
    """
    bound = _under(repository_prefix, identity.workspace)
    return [
        row
        for row in rows
        if row.provenance.job == identity.job_ref
        and row.workspace == identity.workspace
        and row.status == "pending"
        and row.registry == registry
        and _within(row.repository, bound)
    ]


def signing_refusal(
    row: ContainerImage | None, identity: StepIdentity, *, registry: str, repository_prefix: str
) -> str | None:
    """Why ``identity`` may not have ``row`` signed, or None if it may.

    The same entitlement as a credential -- its own job's ``pending`` row, on this registry,
    under its workspace -- because a signature is the stronger grant: it outlives every token,
    and says this system built the image.
    """
    if row is None:
        return f"no such image in workspace {identity.workspace}"
    origin = row.provenance
    if origin.job != identity.job_ref or row.workspace != identity.workspace:
        return f"{row.name} is not job {identity.job_ref}'s"
    if row.status != "pending":
        return f"{row.name} is {row.status}; there is nothing left to sign"
    if row.registry != registry:
        return f"{row.name} is on {row.registry}, not this broker's registry"
    if not _within(row.repository, _under(repository_prefix, identity.workspace)):
        return f"{row.name}'s repository {row.repository!r} is not a repository in workspace {identity.workspace}"
    refusal = _system_tag_refusal(origin.system_tag, identity.workspace)
    return f"{row.name}'s {refusal}" if refusal else None
