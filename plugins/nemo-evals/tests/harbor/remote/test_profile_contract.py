# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frozen sandbox policy and the resolver that copies an operator profile into it."""

import pytest
from nemo_evals.config import HarborTaskJobsConfig
from nemo_evals.harbor.remote.profile import ResolvedHarborSandboxPolicy, ResourceOverrides
from nemo_evals.harbor.remote.sandbox_policy import SandboxPolicyError, resolve_sandbox_policy
from pydantic import ValidationError


def test_policy_round_trips_and_is_frozen(sandbox_policy):
    assert ResolvedHarborSandboxPolicy.model_validate_json(sandbox_policy.model_dump_json()) == sandbox_policy
    with pytest.raises(ValidationError, match="frozen"):
        sandbox_policy.profile = "other"


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"resource_classes": {"gpu": {"template": "harbor-gpu", "dispatcher_profile": "gpu"}}}, "must include 'cpu'"),
        ({"max_sandbox_lifetime_s": 0}, "greater than or equal to 1"),
        ({"surprise": True}, "Extra inputs"),
    ],
)
def test_policy_rejects_invalid_fields(sandbox_policy, update, match):
    with pytest.raises(ValidationError, match=match):
        ResolvedHarborSandboxPolicy.model_validate(sandbox_policy.model_dump() | update)


def test_resource_class_rejects_negative_gpus(sandbox_policy):
    data = sandbox_policy.model_dump()
    data["resource_classes"]["cpu"]["max_gpus_per_task"] = -1
    with pytest.raises(ValidationError):
        ResolvedHarborSandboxPolicy.model_validate(data)


def test_resource_overrides_map_to_harbor_kwargs():
    assert ResourceOverrides(cpus=2, memory_mb=4096).harbor_kwargs() == {"override_cpus": 2, "override_memory_mb": 4096}


def test_resolver_freezes_named_profile_and_platform_server(harbor_config, sandbox_platform):
    policy = resolve_sandbox_policy(harbor_config, sandbox_platform, "harbor-open")
    assert policy.profile == "harbor-open"
    assert policy.egress_allow == ("igw.svc.cluster.local", "inference-api.nvidia.com")
    assert policy.connection.domain == sandbox_platform.sandbox_server_domain
    assert policy.connection.protocol == "http" and policy.connection.use_server_proxy is True
    assert policy.max_sandbox_lifetime_s == 30_600


def test_resolver_freezes_agent_channel_default_empty(harbor_config, sandbox_platform):
    assert resolve_sandbox_policy(harbor_config, sandbox_platform, "harbor-standard").agent_channel == ()
    data = harbor_config.model_dump()
    data["sandbox_profiles"]["harbor-standard"]["agent_channel"] = ["igw.svc.cluster.local"]
    config = HarborTaskJobsConfig.model_validate(data)
    policy = resolve_sandbox_policy(config, sandbox_platform, "harbor-standard")
    assert policy.agent_channel == ("igw.svc.cluster.local",)


def test_resolver_rejects_unknown_profile_listing_names(harbor_config, sandbox_platform):
    with pytest.raises(SandboxPolicyError, match="unknown sandbox_policy 'strict'.*harbor-open, harbor-standard"):
        resolve_sandbox_policy(harbor_config, sandbox_platform, "strict")


@pytest.mark.parametrize("update", [{"sandbox_cluster_capable": False}, {"sandbox_server_domain": None}])
def test_resolver_requires_platform_opensandbox(harbor_config, sandbox_platform, update):
    with pytest.raises(SandboxPolicyError, match="OpenSandbox isn't configured"):
        resolve_sandbox_policy(harbor_config, sandbox_platform.model_copy(update=update), "harbor-standard")


def test_operator_config_requires_digest_pinned_dispatcher_image(harbor_config):
    data = harbor_config.model_dump()
    data["dispatcher_image"] = "registry.invalid/nhx-tasks:latest"
    with pytest.raises(ValidationError, match="dispatcher_image"):
        HarborTaskJobsConfig.model_validate(data)


def test_operator_config_requires_cpu_class(harbor_config):
    data = harbor_config.model_dump()
    data["sandbox_profiles"]["harbor-standard"]["resource_classes"] = {}
    with pytest.raises(ValidationError, match="must include 'cpu'"):
        HarborTaskJobsConfig.model_validate(data)
