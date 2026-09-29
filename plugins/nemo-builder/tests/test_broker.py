# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The broker's rules, with nothing around them: which step a pod is, and what its job may have.

Most of these are refusals. The broker exists so that a step's identity -- reachable by any Editor
through a raw Jobs job -- opens nothing but its own job's destinations.
"""

from __future__ import annotations

import pytest
from nemo_builder_plugin.backend import StepEntitlement
from nemo_builder_plugin.broker import (
    JOB_LABEL,
    MANAGED_BY_JOBS,
    MANAGED_BY_LABEL,
    PROFILE_LABEL,
    STEP_LABEL,
    WORKSPACE_LABEL,
    NotIdentified,
    Pod,
    StepIdentity,
    entitled_rows,
    identify_pod,
    signing_refusal,
)
from nemo_builder_plugin.entities import ContainerImage, Provenance

REGISTRY = "reg.example.com"
ENTITLEMENTS = {
    "push": StepEntitlement(
        step="push", profile="build-push", service_account="nhx-build-push", destinations=("pull", "push"), sign=True
    )
}
IDENTITY = StepIdentity(workspace="ws-a", job="build-1", pod="build-1-push-x", role="push")


def _pod(*, labels: dict[str, str] | None = None, **overrides: str) -> Pod:
    fields = {"name": "build-1-push-x", "uid": "uid-1", "service_account": "nhx-build-push", "phase": "Running"}
    fields |= overrides
    base = {
        MANAGED_BY_LABEL: MANAGED_BY_JOBS,
        WORKSPACE_LABEL: "ws-a",
        JOB_LABEL: "build-1",
        STEP_LABEL: "push",
        PROFILE_LABEL: "build-push",
    }
    return Pod(labels=base | (labels or {}), **fields)


def _identify(pod: Pod, *, uid: str = "uid-1", service_account: str = "nhx-build-push") -> StepIdentity:
    return identify_pod(pod, token_uid=uid, token_service_account=service_account, entitlements=ENTITLEMENTS)


def _row(
    name: str = "build-1-0",
    *,
    workspace: str = "ws-a",
    job: str = "ws-a/build-1",
    repository: str = "ws-a/app",
    registry: str = REGISTRY,
    status: str = "pending",
    system_tag: str = "ws-a--build-1-0",
) -> ContainerImage:
    row = ContainerImage(
        name=name,
        workspace=workspace,
        registry=registry,
        repository=repository,
        provenance=Provenance(
            build_set="build",
            revision=1,
            job=job,
            system_tag=system_tag,
            request_digest="sha256:" + "a" * 64,
        ),
    )
    row.status = status  # ty: ignore[invalid-assignment]
    return row


class TestIdentifyPod:
    def test_a_push_pod_jobs_created_is_its_jobs_push_step(self) -> None:
        assert _identify(_pod()) == IDENTITY
        assert IDENTITY.job_ref == "ws-a/build-1"

    def test_a_pod_whose_kubelet_has_not_reported_running_is_accepted(self) -> None:
        """Measured: a step's first request beats the kubelet's first status update."""
        assert _identify(_pod(phase="Pending")) == IDENTITY

    @pytest.mark.parametrize("phase", ["Succeeded", "Failed", "Unknown"])
    def test_a_finished_pod_is_refused(self, phase: str) -> None:
        """Its token stays valid until the pod object is deleted."""
        with pytest.raises(NotIdentified, match="finished"):
            _identify(_pod(phase=phase))

    def test_a_new_pod_under_an_old_name_is_refused(self) -> None:
        with pytest.raises(NotIdentified, match="not the pod the token was bound to"):
            _identify(_pod(), uid="uid-0")

    def test_a_pod_jobs_did_not_create_is_refused(self) -> None:
        with pytest.raises(NotIdentified, match="not created by Jobs"):
            _identify(_pod(labels={MANAGED_BY_LABEL: "someone"}))

    @pytest.mark.parametrize(
        "labels",
        [{STEP_LABEL: "fetch"}, {PROFILE_LABEL: "build-fetch"}, {STEP_LABEL: "build", PROFILE_LABEL: "build-control"}],
    )
    def test_a_step_no_role_names_is_entitled_to_nothing(self, labels: dict[str, str]) -> None:
        with pytest.raises(NotIdentified, match="entitled to nothing"):
            _identify(_pod(labels=labels))

    def test_the_pod_and_the_token_must_both_name_the_roles_service_account(self) -> None:
        with pytest.raises(NotIdentified, match="the pod names"):
            _identify(_pod(service_account="nhx-build-fetch"))
        with pytest.raises(NotIdentified, match="the token names"):
            _identify(_pod(), service_account="nhx-build-fetch")

    @pytest.mark.parametrize("missing", [WORKSPACE_LABEL, JOB_LABEL])
    def test_a_pod_that_does_not_say_which_job_is_refused(self, missing: str) -> None:
        with pytest.raises(NotIdentified, match="which job"):
            _identify(_pod(labels={missing: ""}))


class TestEntitledRows:
    """The rows only ever narrow the workspace bound; they never widen it."""

    def test_the_jobs_own_pending_rows(self) -> None:
        rows = [_row("build-1-0", repository="ws-a/app"), _row("build-1-1", repository="ws-a/tools")]
        entitled = entitled_rows(rows, IDENTITY, registry=REGISTRY, repository_prefix="")
        assert [row.repository for row in entitled] == ["ws-a/app", "ws-a/tools"]

    def test_another_jobs_rows_are_dropped(self) -> None:
        rows = [_row(job="ws-a/build-2")]
        assert entitled_rows(rows, IDENTITY, registry=REGISTRY, repository_prefix="") == []

    @pytest.mark.parametrize("status", ["ready", "failed"])
    def test_a_settled_row_grants_nothing(self, status: str) -> None:
        """A token for it could only overwrite what was verified."""
        assert entitled_rows([_row(status=status)], IDENTITY, registry=REGISTRY, repository_prefix="") == []

    def test_a_row_on_another_registry_grants_nothing(self) -> None:
        rows = [_row(registry="elsewhere.example.com")]
        assert entitled_rows(rows, IDENTITY, registry=REGISTRY, repository_prefix="") == []

    def test_a_row_naming_a_repository_outside_the_workspace_grants_nothing(self) -> None:
        """Rows are data. One claiming another workspace's path must not widen the grant."""
        rows = [_row(repository="ws-b/app"), _row(repository="ws-ab/app")]
        assert entitled_rows(rows, IDENTITY, registry=REGISTRY, repository_prefix="") == []

    def test_the_bound_includes_the_prefix(self) -> None:
        inside, outside = _row(repository="proj/ws-a/app"), _row(repository="ws-a/app")
        entitled = entitled_rows([inside, outside], IDENTITY, registry=REGISTRY, repository_prefix="proj")
        assert entitled == [inside]

    def test_a_row_from_another_workspace_is_dropped_even_naming_this_job(self) -> None:
        stray = _row(workspace="ws-b", job="ws-a/build-1")
        assert entitled_rows([stray], IDENTITY, registry=REGISTRY, repository_prefix="") == []

    @pytest.mark.parametrize("repository", ["ws-a/x:pull repository:ws-b/app", "ws-a/../ws-b/app", "ws-a/App", "ws-a/"])
    def test_a_repository_that_is_not_a_repository_path_grants_nothing(self, repository: str) -> None:
        """It goes into a token scope as written, where a space asks for a second repository."""
        assert entitled_rows([_row(repository=repository)], IDENTITY, registry=REGISTRY, repository_prefix="") == []


class TestSigningRefusal:
    def test_the_jobs_own_pending_row_may_be_signed(self) -> None:
        assert signing_refusal(_row(), IDENTITY, registry=REGISTRY, repository_prefix="") is None

    @pytest.mark.parametrize(
        ("row", "reason"),
        [
            (None, "no such image"),
            (_row(job="ws-a/build-2"), "is not job ws-a/build-1's"),
            (_row(status="ready"), "nothing left to sign"),
            (_row(registry="elsewhere.example.com"), "not this broker's registry"),
            (_row(repository="ws-b/app"), "not a repository in workspace ws-a"),
            (_row(repository="ws-a/x:pull repository:ws-b/app"), "not a repository in workspace ws-a"),
            # A digest in the tag would have the broker resolve, and sign, one the row's writer chose.
            (_row(system_tag="x@sha256:" + "d" * 64), "not one this system composes"),
            (_row(system_tag="latest"), "not one this system composes"),
            (_row(system_tag="ws-b--build-1-0"), "is not workspace ws-a's"),
        ],
    )
    def test_anything_else_is_refused_with_a_reason(self, row: ContainerImage | None, reason: str) -> None:
        refusal = signing_refusal(row, IDENTITY, registry=REGISTRY, repository_prefix="")
        assert refusal is not None and reason in refusal
