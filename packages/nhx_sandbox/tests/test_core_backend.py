# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared parts of the backend contract: capability checks every backend's ``check`` starts from."""

from __future__ import annotations

import pytest
from nhx_sandbox.core.backend import (
    Capabilities,
    Isolation,
    PlatformEgress,
    Resources,
    SandboxRef,
    SandboxSpec,
    WorkloadNeeds,
    WorkloadUser,
    capability_refusals,
    require_absolute_path,
    require_egress_policy,
    require_own_ref,
)
from nhx_sandbox.core.errors import SandboxLost, SandboxNotFound, UnsupportedOperation
from nhx_sandbox.egress import EgressPolicy
from pydantic import ValidationError

# A backend shaped like a managed service: no root, an assigned uid, the platform's egress, one command at a time.
MANAGED = Capabilities(
    isolation=Isolation.VM,
    exec_as_root=False,
    exec_as_other_users=False,
    workload_user=WorkloadUser.ASSIGNED,
    fixed_uid=None,
    server_ttl=True,
    egress_live_update=False,
    egress_readback=False,
    egress_tld_wildcards=False,
    gpus=False,
    multi_container=False,
    egress_per_sandbox=False,
    concurrent_exec=False,
)


def _caps(**changes: object) -> Capabilities:
    """``MANAGED`` with ``changes``, validated again (``model_copy`` would skip validation)."""
    return Capabilities.model_validate({**MANAGED.model_dump(), **changes})


PER_SANDBOX = _caps(egress_per_sandbox=True)


def test_fixed_uid_is_set_exactly_when_the_uid_is_fixed() -> None:
    _caps(workload_user=WorkloadUser.FIXED, fixed_uid=10001)
    with pytest.raises(ValueError, match="fixed_uid"):
        _caps(workload_user=WorkloadUser.FIXED, fixed_uid=None)
    with pytest.raises(ValueError, match="fixed_uid"):
        _caps(workload_user=WorkloadUser.ASSIGNED, fixed_uid=1000)


def test_a_backend_without_per_sandbox_egress_runs_only_tasks_that_accept_the_platforms_rules() -> None:
    accepting = SandboxSpec(image="i", egress=PlatformEgress(reason="agent egress is the namespace's"))

    assert capability_refusals(MANAGED, accepting, WorkloadNeeds()) == []
    (reason,) = capability_refusals(MANAGED, SandboxSpec(image="i"), WorkloadNeeds())
    assert "PlatformEgress" in reason


def test_a_backend_with_per_sandbox_egress_refuses_the_platforms_rules() -> None:
    accepting = SandboxSpec(image="i", egress=PlatformEgress(reason="r"))

    assert capability_refusals(PER_SANDBOX, SandboxSpec(image="i"), WorkloadNeeds()) == []
    assert capability_refusals(PER_SANDBOX, accepting, WorkloadNeeds())
    with pytest.raises(UnsupportedOperation):
        require_egress_policy(accepting, "test")
    assert require_egress_policy(SandboxSpec(image="i"), "test") == EgressPolicy()


def test_every_need_the_backend_cant_meet_is_named() -> None:
    spec = SandboxSpec(image="i", resources=Resources(gpus=1))
    needs = WorkloadNeeds(runs_as_root=True, multi_container=True)

    reasons = capability_refusals(PER_SANDBOX, spec, needs)

    assert len(reasons) == 3
    assert any("GPU" in reason for reason in reasons)
    assert any("root" in reason for reason in reasons)
    assert any("containers" in reason for reason in reasons)


def test_a_lost_sandbox_is_one_that_is_no_longer_found() -> None:
    """Code that already handles a vanished sandbox handles a lost one the same way."""
    assert issubclass(SandboxLost, SandboxNotFound)


def test_isolation_levels_compare_weakest_first() -> None:
    assert Isolation.VM.at_least(Isolation.CONTAINER)
    assert Isolation.CONTAINER.at_least(Isolation.CONTAINER)
    assert not Isolation.NONE.at_least(Isolation.CONTAINER)


def test_a_ref_from_another_backend_is_refused() -> None:
    """Ids are only unique within one backend, so acting on another backend's ref could hit the wrong sandbox."""
    require_own_ref(SandboxRef(backend="local", id="1", name="a"), "local")
    with pytest.raises(ValueError, match="belongs to the openshell backend"):
        require_own_ref(SandboxRef(backend="openshell", id="1", name="a"), "local")


def test_sandbox_paths_must_be_absolute() -> None:
    require_absolute_path("/workspace/out.txt")
    with pytest.raises(ValueError, match="absolute"):
        require_absolute_path("out.txt")


def test_specs_are_checked_where_they_are_built() -> None:
    """Specs come from task configs too, so a typo or an impossible value fails at construction, not in a backend."""
    with pytest.raises(ValidationError):
        SandboxSpec.model_validate({"image": "i", "ttl": 60})
    with pytest.raises(ValidationError):
        SandboxSpec(image="i", ttl_s=-1)
    spec = SandboxSpec.model_validate({"image": "i", "egress": {"reason": "agent egress is the namespace's"}})
    assert isinstance(spec.egress, PlatformEgress)
