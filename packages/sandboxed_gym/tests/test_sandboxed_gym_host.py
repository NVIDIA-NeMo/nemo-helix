# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ipaddress

import pytest
from sandboxed_gym import egress
from sandboxed_gym.egress import _as_network, build_egress_policy, denied_cidrs
from sandboxed_gym.host.models import (
    GymHostEgressRule,
    GymHostSpec,
    GymHostVolumeMount,
    NemoGymSandboxedConfig,
    SandboxConfig,
    SandboxNetworkPolicy,
    build_bootstrap_env,
    validate_bootstrap_env,
)
from sandboxed_gym.wire import BROKER_TOKEN_ENV, BROKER_URL_ENV


def _mount(claim: str, path: str, read_only: bool) -> GymHostVolumeMount:
    return GymHostVolumeMount(
        pvc_claim=claim,
        mount_path=path,
        read_only=read_only,
    )


def test_bootstrap_env_rejects_opensandbox_credentials():
    with pytest.raises(ValueError, match="OPENSANDBOX_API_KEY"):
        validate_bootstrap_env({"OPENSANDBOX_API_KEY": "secret"})


def test_build_bootstrap_env_sets_required_keys():
    env = build_bootstrap_env(
        "job-1",
        "/job/environment",
        "/job/work",
        "http://broker:51234",
        "token",
        1024,
        2048,
        dataset_path="/job/dataset",
    )
    assert env["NMP_JOB_ID"] == "job-1"
    assert env["NMP_BROKER_URL"] == "http://broker:51234"
    assert env["NMP_DATASET_PATH"] == "/job/dataset"
    assert "OPENSANDBOX_API_KEY" not in env
    assert env[BROKER_URL_ENV] == "http://broker:51234"
    assert env[BROKER_TOKEN_ENV] == "token"


def test_gym_host_spec_enforces_mount_roles():
    with pytest.raises(ValueError, match="environment_mount must be read-only"):
        GymHostSpec(
            job_id="job-1",
            runtime_image="runtime:dev",
            environment_mount=_mount("claim", "/job/environment", False),
            workspace_mount=_mount("claim", "/job/work", False),
        )


def test_egress_rule_accepts_ip_or_fqdn():
    assert GymHostEgressRule(host="10.0.0.5", port=8000).host == "10.0.0.5"
    assert GymHostEgressRule(host="vllm.svc.cluster.local", port=8000).host == ("vllm.svc.cluster.local")
    with pytest.raises(ValueError, match="blank"):
        GymHostEgressRule(host="  ", port=8000)


def _denies(policy):
    return {rule.target for rule in policy.rules if rule.action == "deny"}


def _allows(policy):
    return {rule.target for rule in policy.rules if rule.action == "allow"}


def _is_denied(address, denied_targets):
    addr = ipaddress.ip_address(address)
    return any(
        addr in ipaddress.ip_network(target)
        for target in denied_targets
        if ipaddress.ip_network(target).version == addr.version
    )


def test_host_egress_policy_allow_internet_is_still_default_deny():
    spec = GymHostSpec(
        job_id="job-1",
        runtime_image="runtime:dev",
        environment_mount=_mount("claim", "/job/environment", True),
        workspace_mount=_mount("claim", "/job/work", False),
        egress_allow=(GymHostEgressRule(host="vllm.svc.cluster.local", port=8000),),
        allow_internet=True,
    )
    policy = build_egress_policy(spec.egress_allowlist)
    assert policy.default_action == "deny"
    assert {"vllm.svc.cluster.local", "*.com", "*.org"} <= _allows(policy)
    # Public space is reached by resolving an allowed name, never by allowing the address range:
    # an allowed public CIDR would let the sandbox dial any host in it without asking the resolver.
    networks = (_as_network(target) for target in _allows(policy))
    assert not any(network is not None and not network.is_private for network in networks)


def test_host_egress_policy_without_internet_has_no_dns_suffixes():
    spec = GymHostSpec(
        job_id="job-1",
        runtime_image="runtime:dev",
        environment_mount=_mount("claim", "/job/environment", True),
        workspace_mount=_mount("claim", "/job/work", False),
        egress_allow=(GymHostEgressRule(host="broker.svc.cluster.local", port=51234),),
        allow_internet=False,
        public_dns_allow=("*.io",),
        resolver_addresses=(),
    )
    policy = build_egress_policy(spec.egress_allowlist)
    assert policy.default_action == "deny"
    assert _allows(policy) == {"broker.svc.cluster.local"}
    # Deny rules are not conditional on allow_internet: a sandbox can come to hold a private
    # address through a name it was allowed for other reasons.
    assert _is_denied("10.0.0.1", _denies(policy))


def test_host_egress_policy_adds_custom_public_dns_suffixes():
    spec = GymHostSpec(
        job_id="job-1",
        runtime_image="runtime:dev",
        environment_mount=_mount("claim", "/job/environment", True),
        workspace_mount=_mount("claim", "/job/work", False),
        allow_internet=True,
        public_dns_allow=("*.io",),
    )
    assert {"*.com", "*.org", "*.io"} <= _allows(build_egress_policy(spec.egress_allowlist))


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "100.64.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "192.168.0.1",
        "198.18.0.1",
        "224.0.0.1",
        "240.0.0.1",
        "fd00::1",
        "fe80::1",
    ],
)
def test_denied_cidrs_cover_non_public_space(address):
    assert _is_denied(address, denied_cidrs(resolver_addresses=()))


@pytest.mark.parametrize("address", ["127.0.0.1", "127.0.0.53", "::1"])
def test_denied_cidrs_leave_loopback_alone(address):
    """Loopback is the sandbox's own namespace, and the egress sidecar's DNS proxy lives there.

    All port-53 traffic is redirected to that proxy on ``127.0.0.1``, and deny beats allow, so a
    denied loopback drops every lookup before it can reach the resolver.
    """
    assert not _is_denied(address, denied_cidrs(resolver_addresses=()))


def test_denied_cidrs_leave_public_space_reachable():
    """A deny that swallowed public space would make the DNS whitelist useless."""
    denied = denied_cidrs(resolver_addresses=())
    assert not _is_denied("8.8.8.8", denied)
    assert not _is_denied("2606:4700::1111", denied)
    assert "0.0.0.0/0" not in denied


def test_denied_cidrs_carve_out_an_allowed_private_address():
    """The whole point: an allowed host inside a denied range must survive the deny.

    OpenSandbox evaluates ``@deny`` before both the DNS-learned set and ``@allow``, so a denied
    ``10.0.0.0/8`` would beat an allowed ``10.0.0.51`` rather than lose to it.
    """
    denied = denied_cidrs(("10.0.0.51",), resolver_addresses=())
    assert not _is_denied("10.0.0.51", denied)
    assert _is_denied("10.0.0.50", denied)
    assert _is_denied("10.0.0.52", denied)
    assert _is_denied("10.255.255.255", denied)


def test_denied_cidrs_split_repeatedly_for_several_allowed_addresses():
    """Each further allowed address splits whichever fragment now contains it."""
    allowed = ("10.0.0.51", "10.0.0.83", "10.0.0.193")
    denied = denied_cidrs(allowed, resolver_addresses=())
    for address in allowed:
        assert not _is_denied(address, denied)
    for address in ("10.0.0.50", "10.0.0.84", "10.0.0.192", "10.0.0.194", "10.1.0.1"):
        assert _is_denied(address, denied)
    # Fragments must stay disjoint; nftables interval sets reject overlapping elements.
    networks = sorted(ipaddress.ip_network(target) for target in denied if ":" not in target)
    for earlier, later in zip(networks, networks[1:]):
        assert earlier.broadcast_address < later.network_address


def test_denied_cidrs_carve_out_a_resolvable_hostname(monkeypatch):
    """Hostnames are resolved so a private endpoint named by DNS is not denied by address."""
    monkeypatch.setattr(egress, "_resolve_host", lambda target: (ipaddress.ip_network("10.0.0.51"),))
    denied = denied_cidrs(("private.svc.cluster.local",), resolver_addresses=())
    assert not _is_denied("10.0.0.51", denied)
    assert _is_denied("10.0.0.52", denied)


def test_denied_cidrs_ignore_wildcard_dns_targets():
    assert denied_cidrs(("*.com",), resolver_addresses=()) == denied_cidrs(resolver_addresses=())


def test_denied_cidrs_carve_out_the_resolver():
    """A denied resolver drops every lookup, so nothing resolves and no allow rule can fire."""
    denied = denied_cidrs(resolver_addresses=("10.96.5.5",))
    assert not _is_denied("10.96.5.5", denied)
    assert _is_denied("10.96.5.6", denied)


def test_host_egress_policy_explicitly_allows_the_resolver():
    spec = GymHostSpec(
        job_id="job-1",
        runtime_image="runtime:dev",
        environment_mount=_mount("claim", "/job/environment", True),
        workspace_mount=_mount("claim", "/job/work", False),
        resolver_addresses=("10.96.5.5",),
    )
    policy = build_egress_policy(spec.egress_allowlist)
    assert "10.96.5.5" in _allows(policy)
    assert not _is_denied("10.96.5.5", _denies(policy))


def test_denied_cidrs_default_the_resolver_to_the_local_nameservers(tmp_path, monkeypatch):
    resolv_conf = tmp_path / "resolv.conf"
    resolv_conf.write_text("search svc.cluster.local\nnameserver 10.96.5.5  # cluster\n")
    monkeypatch.setattr(egress, "RESOLV_CONF_PATH", str(resolv_conf))

    assert egress.local_resolver_addresses() == ("10.96.5.5",)
    assert not _is_denied("10.96.5.5", denied_cidrs())


def test_local_resolver_addresses_tolerate_a_missing_resolv_conf(tmp_path, monkeypatch):
    monkeypatch.setattr(egress, "RESOLV_CONF_PATH", str(tmp_path / "absent"))
    assert egress.local_resolver_addresses() == ()


def test_sandboxed_config_requires_sandbox_block():
    with pytest.raises(ValueError, match="sandbox"):
        NemoGymSandboxedConfig(sandboxed=True)


def _serve_cfg(gym_global_config=None, **overrides):
    from sandboxed_gym.serve_config import SandboxedGymServeConfig

    data = {
        "job_id": "job-1",
        "sandbox": {
            "image": "runtime:dev",
            "network_policy": {"egress_allow": []},
            "environment_pvc_claim": "env-pvc",
            "workspace_pvc_claim": "work-pvc",
        },
        "gym_global_config": gym_global_config or {},
        "policy_base_urls": ["http://vllm-0.svc.cluster.local:8000/v1"],
    }
    data.update(overrides)
    return SandboxedGymServeConfig.model_validate(data)


def test_sandbox_runtime_defaults_preserve_caller_gym_config():
    from sandboxed_gym.orchestrator import apply_sandbox_runtime_defaults

    global_config = apply_sandbox_runtime_defaults(
        {
            "config_paths": ["/job/environment/nemo-environment.yaml"],
            "policy_model_name": "meta-llama/Llama-3.1-8B",
            "policy_base_url": ["http://vllm-0.svc.cluster.local:8000/v1"],
            "ray_head_node_address": "10.0.0.5:6379",
        }
    )

    assert global_config["config_paths"] == ["/job/environment/nemo-environment.yaml"]
    assert "ray_head_node_address" not in global_config
    assert global_config["policy_model_name"] == "meta-llama/Llama-3.1-8B"
    assert global_config["policy_base_url"] == ["http://vllm-0.svc.cluster.local:8000/v1"]
    assert global_config["default_host"] == "127.0.0.1"
    assert global_config["port_range_low"] < global_config["port_range_high"]
    assert global_config["uv_cache_dir"]
    assert global_config["uv_venv_dir"]


def test_sandbox_runtime_defaults_target_gym_installs_at_the_activated_venv():
    """Gym must install into the venv it activated, not whatever UV_PYTHON names.

    The training image sets UV_PYTHON to an absolute interpreter path, which outranks the active
    venv when Gym's ``uv pip install`` resolves a target. Without this the install lands in a
    read-only tree and every Gym server dies during startup.
    """
    from sandboxed_gym.orchestrator import apply_sandbox_runtime_defaults

    assert apply_sandbox_runtime_defaults({})["uv_pip_set_python"] is True


def test_sandbox_runtime_defaults_respect_an_explicit_uv_pip_set_python():
    """A caller that has its own answer keeps it -- every uv key here is a default, not an override."""
    from sandboxed_gym.orchestrator import apply_sandbox_runtime_defaults

    assert apply_sandbox_runtime_defaults({"uv_pip_set_python": False})["uv_pip_set_python"] is False


def test_gym_host_spec_defers_to_the_image_entrypoint_when_omitted():
    # The orchestrator cannot name a path inside the host image: resolving one from its own
    # installation would describe its own container. With no entrypoint configured the image
    # starts itself through its CMD.
    from sandboxed_gym.config import BrokerEndpoint
    from sandboxed_gym.orchestrator import build_gym_host_spec

    cfg = _serve_cfg()
    assert cfg.sandbox.entrypoint is None
    broker = BrokerEndpoint(
        url="http://broker.svc.cluster.local:51234",
        host="broker.svc.cluster.local",
        port=51234,
        token="token",
    )
    spec = build_gym_host_spec(cfg, broker)
    assert spec.entrypoint is None


def test_gym_host_spec_carries_a_configured_entrypoint_through():
    from sandboxed_gym.config import BrokerEndpoint
    from sandboxed_gym.orchestrator import build_gym_host_spec

    cfg = _serve_cfg(
        sandbox={
            "image": "runtime:dev",
            "network_policy": {"egress_allow": []},
            "environment_pvc_claim": "env-pvc",
            "workspace_pvc_claim": "work-pvc",
            "entrypoint": ["/bin/sh", "/opt/nemo-rl/gym_host.sh"],
        }
    )
    broker = BrokerEndpoint(
        url="http://broker.svc.cluster.local:51234",
        host="broker.svc.cluster.local",
        port=51234,
        token="token",
    )
    spec = build_gym_host_spec(cfg, broker)
    assert spec.entrypoint == ("/bin/sh", "/opt/nemo-rl/gym_host.sh")


def test_gym_host_spec_forwards_the_rollout_deadline():
    """The host has to know when to give up, and only the caller's config knows when that is.

    Without this the host falls back to its own 30-minute default while the client gives up at
    ``rollout_timeout_s``, so a stuck batch never produces the clean ``deadline_exceeded`` body
    the client timeout was supposed to replace.
    """
    from sandboxed_gym.config import BrokerEndpoint
    from sandboxed_gym.orchestrator import build_gym_host_spec
    from sandboxed_gym.serve_config import SandboxedGymServeConfig

    cfg = SandboxedGymServeConfig.model_validate(
        {
            "job_id": "job-1",
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
                "rollout_timeout_s": 120,
            },
        }
    )
    broker = BrokerEndpoint(url="http://broker:1", host="broker", port=1, token="t")

    spec = build_gym_host_spec(cfg, broker)

    assert spec.bootstrap_env["NMP_ROLLOUT_DEADLINE_S"] == "120.0"


def test_gym_host_spec_carries_global_config_in_bootstrap():
    import json

    from sandboxed_gym.config import BrokerEndpoint
    from sandboxed_gym.orchestrator import build_gym_host_spec
    from sandboxed_gym.runtime.gym_host_runtime import GYM_GLOBAL_CONFIG_ENV_KEY

    cfg = _serve_cfg({"config_paths": ["/job/environment/env.yaml"]})
    broker = BrokerEndpoint(
        url="http://broker.svc.cluster.local:51234",
        host="broker.svc.cluster.local",
        port=51234,
        token="token",
    )
    spec = build_gym_host_spec(cfg, broker)

    embedded = json.loads(spec.bootstrap_env[GYM_GLOBAL_CONFIG_ENV_KEY])
    assert embedded["config_paths"] == ["/job/environment/env.yaml"]
    assert any(rule.host == "broker.svc.cluster.local" for rule in spec.egress_allow)


def test_explicit_environment_path_requires_a_valid_package():
    from sandboxed_gym.config import BrokerEndpoint
    from sandboxed_gym.orchestrator import build_gym_host_spec
    from sandboxed_gym.runtime.gym_host_runtime import ENVIRONMENT_PACKAGE_REQUIRED_ENV_KEY

    broker = BrokerEndpoint(
        url="http://broker.svc.cluster.local:51234",
        host="broker.svc.cluster.local",
        port=51234,
        token="token",
    )

    custom = build_gym_host_spec(_serve_cfg(environment_path="/job/environment"), broker)
    bundled = build_gym_host_spec(_serve_cfg(), broker)

    assert custom.bootstrap_env[ENVIRONMENT_PACKAGE_REQUIRED_ENV_KEY] == "true"
    assert ENVIRONMENT_PACKAGE_REQUIRED_ENV_KEY not in bundled.bootstrap_env


def test_sandbox_config_round_trip():
    cfg = SandboxConfig(
        image="runtime:dev",
        network_policy=SandboxNetworkPolicy(egress_allow=[GymHostEgressRule(host="vllm.svc", port=8000)]),
        environment_pvc_claim="env-pvc",
        workspace_pvc_claim="work-pvc",
    )
    assert cfg.env_mount_path == "/job/environment"
    assert cfg.network_policy.egress_allow[0].host == "vllm.svc"
    assert cfg.network_policy.public_dns_allow is None


def test_collect_gym_host_egress_allows_discovers_and_dedupes():
    from sandboxed_gym.orchestrator import collect_gym_host_egress_allows

    rules = collect_gym_host_egress_allows(
        configured=[
            GymHostEgressRule(host="vllm.svc", port=8000),
            GymHostEgressRule(host="broker.svc", port=51234),
        ],
        broker_host="broker.svc",
        broker_port=51234,
        base_urls=[
            "http://vllm.svc:8000/v1",
            "https://64.181.219.176/v1",
        ],
    )

    assert [(rule.host, rule.port) for rule in rules] == [
        ("vllm.svc", 8000),
        ("broker.svc", 51234),
        ("64.181.219.176", 443),
    ]


def test_uv_env_passthrough_forwards_only_set_vars(monkeypatch):
    from sandboxed_gym.host.models import uv_env_passthrough

    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)

    assert uv_env_passthrough() == {
        "NRL_CONTAINER": "1",
        "NEMO_GYM_VENV_DIR": "/opt/gym_venvs",
    }


def test_uv_env_passthrough_carries_no_credentials(monkeypatch):
    """Whatever it forwards must survive validate_bootstrap_env."""
    from sandboxed_gym.host.models import (
        uv_env_passthrough,
        validate_bootstrap_env,
    )

    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("UV_CACHE_DIR", "/opt/uv_cache")
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")

    validate_bootstrap_env(uv_env_passthrough())


def _offline_spec(*, environment_offline: bool):
    from sandboxed_gym.config import BrokerEndpoint
    from sandboxed_gym.orchestrator import build_gym_host_spec
    from sandboxed_gym.serve_config import SandboxedGymServeConfig

    cfg = SandboxedGymServeConfig.model_validate(
        {
            "job_id": "job-1",
            "environment_offline": environment_offline,
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
            },
        }
    )
    broker = BrokerEndpoint(url="http://broker:1", host="broker", port=1, token="t")
    return build_gym_host_spec(cfg, broker)


def test_gym_host_spec_forwards_an_offline_environment():
    """Only the caller who built the package knows the wheelhouse is a complete closure.

    The sandbox has no way to find out: a wheels-v1 package can ship wheels and still need an
    index for its agent, so without this flag the host leaves uv's index fallback in place and a
    configured-but-unreachable index fails the per-component venvs.
    """
    from sandboxed_gym.runtime.gym_host_runtime import ENVIRONMENT_OFFLINE_ENV_KEY

    spec = _offline_spec(environment_offline=True)

    assert spec.bootstrap_env[ENVIRONMENT_OFFLINE_ENV_KEY] == "true"


def test_gym_host_spec_leaves_the_index_alone_by_default():
    from sandboxed_gym.runtime.gym_host_runtime import ENVIRONMENT_OFFLINE_ENV_KEY

    spec = _offline_spec(environment_offline=False)

    assert ENVIRONMENT_OFFLINE_ENV_KEY not in spec.bootstrap_env


def test_the_caller_dialect_accepts_an_offline_environment():
    """`env.nemo_gym` is `extra="forbid"`, so a key the platform emits and this model lacks is
    not ignored -- it fails validation. nmp/rl's grpo_config sets `environment_offline` on an
    offline manifest, so without the field a sandboxed run with one cannot start at all."""
    from sandboxed_gym.host.models import NemoGymSandboxedConfig

    cfg = NemoGymSandboxedConfig.model_validate(
        {
            "sandboxed": True,
            "environment_path": "/job/environment",
            "environment_offline": True,
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
            },
        }
    )

    assert cfg.environment_offline is True
