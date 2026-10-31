# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frozen sandbox policy and the resolver that copies an operator profile into it."""

import pytest
from harbor.models.task.config import TaskConfig
from nemo_evals.config import HarborTaskJobsConfig
from nemo_evals.harbor.remote.profile import (
    ResolvedHarborSandboxPolicy,
    ResourceOverrides,
    SandboxSelection,
    resource_defaults,
)
from nemo_evals.harbor.remote.resources import defaulted_resources_notice, task_sandboxes
from nemo_evals.harbor.remote.sandbox_policy import SandboxPolicyError, resolve_sandbox_policy, task_resources_violation
from pydantic import ValidationError


def test_policy_round_trips_and_is_frozen(sandbox_policy):
    assert ResolvedHarborSandboxPolicy.model_validate_json(sandbox_policy.model_dump_json()) == sandbox_policy
    with pytest.raises(ValidationError, match="frozen"):
        sandbox_policy.profile = "other"


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"resource_classes": {"cpu": {"template": "harbor-cpu", "dispatcher_profile": "k8s-cpu"}}}, "Extra inputs"),
        ({"root_template": "harbor-cpu-root"}, "Extra inputs"),
        ({"max_sandbox_lifetime_s": 0}, "greater than or equal to 1"),
        ({"surprise": True}, "Extra inputs"),
    ],
)
def test_policy_rejects_invalid_fields(sandbox_policy, update, match):
    with pytest.raises(ValidationError, match=match):
        ResolvedHarborSandboxPolicy.model_validate(sandbox_policy.model_dump() | update)


def test_resource_defaults_are_capped_by_the_ceilings(sandbox_policy):
    assert resource_defaults(sandbox_policy.model_copy(update={"max_cpus": None, "max_memory_mb": None})) == {
        "cpus": 1,
        "memory_mb": 2048,
    }
    capped = sandbox_policy.model_copy(update={"max_cpus": 8, "max_memory_mb": 1024})
    assert resource_defaults(capped) == {"cpus": 1, "memory_mb": 1024}


SIZED = {"cpus": 2, "memory_mb": 4096}


@pytest.mark.parametrize(
    ("config", "overrides", "ceilings", "violation"),
    [
        ({"environment": {**SIZED, "cpus": 16}}, {}, {"max_cpus": 8}, "agent sandbox cpus=16 (task.toml)"),
        ({"environment": {**SIZED, "cpus": 8}}, {}, {"max_cpus": 8}, None),
        ({"environment": {**SIZED, "cpus": 16}}, {"cpus": 4}, {"max_cpus": 8}, None),
        ({"environment": SIZED}, {"cpus": 16}, {}, None),  # an override over the ceiling never gets this far
        (
            {"environment": SIZED, "verifier": {"environment_mode": "separate", "environment": {"memory_mb": 9000}}},
            {},
            {"max_memory_mb": 8192},
            "verifier sandbox memory_mb=9000 (task.toml)",
        ),
        (
            {"environment": {**SIZED, "memory_mb": 9000}, "verifier": {}},
            {"memory_mb": 4096},
            {"max_memory_mb": 8192},
            None,
        ),
        ({"environment": {}}, {}, {"max_cpus": 1, "max_memory_mb": 1024}, None),  # the defaults are capped
        ({"environment": {**SIZED, "storage_mb": 50_000}}, {}, {"max_storage_mb": 40_000}, "agent sandbox storage_mb"),
        ({"environment": {**SIZED, "storage_mb": 50_000}}, {}, {"max_disk_mb": 40_000}, None),  # clamped, not refused
        ({"environment": {"cpus": 64, "memory_mb": 1_000_000}}, {}, {}, None),
    ],
)
def test_task_resources_violation(sandbox_policy, config, overrides, ceilings, violation):
    policy = sandbox_policy.model_copy(
        update={
            "max_cpus": None,
            "max_memory_mb": None,
            "max_storage_mb": None,
            **ceilings,
            "resource_overrides": ResourceOverrides(**overrides),
        }
    )
    found = task_resources_violation(TaskConfig.model_validate(config), policy)
    if violation is None:
        assert found is None
    else:
        assert found is not None and found.startswith(violation) and policy.profile in found


@pytest.mark.parametrize("enforce_disk_limit", [True, False])
def test_a_storage_only_task_gets_default_cpu_and_memory(sandbox_policy, enforce_disk_limit):
    policy = sandbox_policy.model_copy(update={"enforce_disk_limit": enforce_disk_limit, "max_memory_mb": 1024})
    config = TaskConfig.model_validate(
        {"environment": {"storage_mb": 4096}, "verifier": {"environment_mode": "separate"}}
    )
    for sandbox in task_sandboxes(config, policy):
        assert (sandbox.config.cpus, sandbox.config.memory_mb, sandbox.config.storage_mb) == (1, 1024, 4096)
        assert sandbox.defaulted == ("cpus", "memory_mb")
    assert config.environment.cpus is None  # never changed in place


def test_defaulted_resources_notice_names_each_sandbox(sandbox_policy):
    policy = sandbox_policy.model_copy(update={"max_cpus": None, "max_memory_mb": 1024})
    tasks = [
        ("hello/a", {"environment": {"cpus": 2}}),
        ("hello/b", {"environment": SIZED, "verifier": {"environment_mode": "separate", "environment": {}}}),
        ("hello/c", {"environment": SIZED}),
    ]
    assert defaulted_resources_notice(tasks, policy) == (
        "2 tasks rely on default sandbox resources: hello/a agent memory_mb=1024, "
        "hello/b verifier cpus=1 memory_mb=1024. Set them in task.toml or pass --cpus/--memory-mb."
    )
    assert defaulted_resources_notice(tasks[2:], policy) is None


def test_omitted_selection_freezes_the_default_profile(harbor_config, sandbox_platform):
    for selection in (None, SandboxSelection(), SandboxSelection(cpus=2)):
        policy = resolve_sandbox_policy(harbor_config, sandbox_platform, selection)
        assert policy.profile == harbor_config.default_sandbox_profile == "harbor-standard"
    config = harbor_config.model_copy(update={"default_sandbox_profile": "harbor-open"})
    assert resolve_sandbox_policy(config, sandbox_platform).profile == "harbor-open"


def test_selected_profile_name_is_not_an_override(harbor_config, sandbox_platform):
    policy = resolve_sandbox_policy(harbor_config, sandbox_platform, SandboxSelection(profile="harbor-open", cpus=2))
    assert policy.resource_overrides == ResourceOverrides(cpus=2)


def test_resolver_freezes_named_profile_and_platform_server(harbor_config, sandbox_platform):
    policy = resolve_sandbox_policy(harbor_config, sandbox_platform, SandboxSelection(profile="harbor-open"))
    assert policy.profile == "harbor-open"
    assert policy.egress_allow == ("igw.svc.cluster.local", "inference-api.nvidia.com")
    assert policy.connection.domain == sandbox_platform.sandbox_server_domain
    assert policy.connection.protocol == "http" and policy.connection.use_server_proxy is True
    assert policy.max_sandbox_lifetime_s == 30_600


def test_resolver_freezes_agent_channel_default_empty(harbor_config, sandbox_platform):
    assert resolve_sandbox_policy(harbor_config, sandbox_platform).agent_channel == ()
    data = harbor_config.model_dump()
    data["sandbox_profiles"]["harbor-standard"]["agent_channel"] = ["igw.svc.cluster.local"]
    config = HarborTaskJobsConfig.model_validate(data)
    policy = resolve_sandbox_policy(config, sandbox_platform)
    assert policy.agent_channel == ("igw.svc.cluster.local",)


def test_resolver_rejects_unknown_profile_listing_names(harbor_config, sandbox_platform):
    with pytest.raises(
        SandboxPolicyError, match="unknown execution.sandbox.profile 'strict'.*harbor-open, harbor-standard"
    ):
        resolve_sandbox_policy(harbor_config, sandbox_platform, SandboxSelection(profile="strict"))


@pytest.mark.parametrize("update", [{"sandbox_cluster_capable": False}, {"sandbox_server_domain": None}])
def test_resolver_requires_platform_opensandbox(harbor_config, sandbox_platform, update):
    with pytest.raises(SandboxPolicyError, match="OpenSandbox isn't configured"):
        resolve_sandbox_policy(harbor_config, sandbox_platform.model_copy(update=update))


def test_operator_config_requires_digest_pinned_dispatcher_image(harbor_config):
    data = harbor_config.model_dump()
    data["dispatcher_image"] = "registry.invalid/nhx-tasks:latest"
    with pytest.raises(ValidationError, match="dispatcher_image"):
        HarborTaskJobsConfig.model_validate(data)


def test_operator_config_may_omit_dispatcher_image(harbor_config):
    data = harbor_config.model_dump()
    del data["dispatcher_image"]
    assert HarborTaskJobsConfig.model_validate(data).dispatcher_image is None


def test_operator_config_requires_the_default_profile(harbor_config):
    data = harbor_config.model_dump()
    data["default_sandbox_profile"] = "harbor-gpu"
    with pytest.raises(ValidationError, match="default_sandbox_profile 'harbor-gpu' isn't a sandbox_profiles key"):
        HarborTaskJobsConfig.model_validate(data)


@pytest.mark.parametrize(
    "removed",
    [
        {"resource_classes": {"cpu": {"template": "harbor-cpu", "dispatcher_profile": "k8s-cpu"}}},
        {"root_template": "x"},
    ],
)
def test_operator_config_rejects_removed_template_fields(harbor_config, removed):
    """A deployment config still setting them fails at startup rather than seeming to select a template."""
    data = harbor_config.model_dump()
    data["sandbox_profiles"]["harbor-standard"].update(removed)
    with pytest.raises(ValidationError, match="Extra inputs"):
        HarborTaskJobsConfig.model_validate(data)


def with_profile(harbor_config: HarborTaskJobsConfig, **settings: object) -> HarborTaskJobsConfig:
    data = harbor_config.model_dump()
    data["sandbox_profiles"]["harbor-standard"].update(settings)
    return HarborTaskJobsConfig.model_validate(data)


def test_resolver_freezes_request_overrides(harbor_config, sandbox_platform):
    overrides = ResourceOverrides(cpus=4, memory_mb=8192)
    policy = resolve_sandbox_policy(harbor_config, sandbox_platform, SandboxSelection(**overrides.model_dump()))
    assert policy.resource_overrides == overrides
    assert resolve_sandbox_policy(harbor_config, sandbox_platform).resource_overrides == (ResourceOverrides())


def test_resolver_freezes_ceilings_and_disk_limit(harbor_config, sandbox_platform):
    default = resolve_sandbox_policy(harbor_config, sandbox_platform)
    assert (default.max_cpus, default.max_memory_mb, default.max_storage_mb) == (None, None, None)
    assert default.enforce_disk_limit is True and default.max_disk_mb == 131_072
    config = with_profile(harbor_config, max_cpus=8, max_memory_mb=16_384, enforce_disk_limit=False, max_disk_mb=None)
    policy = resolve_sandbox_policy(config, sandbox_platform)
    assert (policy.max_cpus, policy.max_memory_mb) == (8, 16_384)
    assert policy.enforce_disk_limit is False and policy.max_disk_mb is None


def test_resolver_leaves_enforce_closed_book_out_of_the_policy(harbor_config, sandbox_platform):
    assert harbor_config.sandbox_profiles["harbor-standard"].enforce_closed_book is False
    policy = resolve_sandbox_policy(with_profile(harbor_config, enforce_closed_book=True), sandbox_platform)
    assert "enforce_closed_book" not in policy.model_dump()


@pytest.mark.parametrize(
    ("field", "ceiling"), [("cpus", "max_cpus"), ("memory_mb", "max_memory_mb"), ("storage_mb", "max_storage_mb")]
)
def test_resolver_rejects_overrides_over_the_profile_ceiling(harbor_config, sandbox_platform, field, ceiling):
    config = with_profile(harbor_config, **{ceiling: 1000})
    at_ceiling = resolve_sandbox_policy(config, sandbox_platform, SandboxSelection.model_validate({field: 1000}))
    assert getattr(at_ceiling.resource_overrides, field) == 1000
    with pytest.raises(SandboxPolicyError, match=rf"execution.sandbox.{field}=1001 exceeds .*'s {ceiling}=1000"):
        resolve_sandbox_policy(config, sandbox_platform, SandboxSelection.model_validate({field: 1001}))


def test_resolver_without_a_ceiling_accepts_any_override(harbor_config, sandbox_platform):
    overrides = ResourceOverrides(cpus=512, memory_mb=1_048_576)
    assert resolve_sandbox_policy(
        harbor_config, sandbox_platform, SandboxSelection(**overrides.model_dump())
    ).resource_overrides == (overrides)


def test_resolver_caps_storage_at_the_disk_limit(harbor_config, sandbox_platform):
    resolve_sandbox_policy(harbor_config, sandbox_platform, SandboxSelection(storage_mb=131_072))
    with pytest.raises(SandboxPolicyError, match="storage_mb=131073 exceeds .*'s max_disk_mb=131072"):
        resolve_sandbox_policy(harbor_config, sandbox_platform, SandboxSelection(storage_mb=131_073))
    unenforced = with_profile(harbor_config, enforce_disk_limit=False)
    resolve_sandbox_policy(unenforced, sandbox_platform, SandboxSelection(storage_mb=262_144))
