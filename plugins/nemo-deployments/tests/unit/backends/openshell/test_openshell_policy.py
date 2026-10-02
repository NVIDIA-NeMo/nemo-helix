# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the policy YAML -> SandboxPolicy proto builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from nemo_deployments_plugin.backends.openshell.policy import (
    PLATFORM_EGRESS_KEY,
    HelixEgress,
    SandboxFilesystem,
    build_sandbox_policy,
    generate_policy_dict,
    generate_sandbox_policy,
    inject_platform_egress,
    load_sandbox_policy,
    normalize_loaded_policy,
)

# Building a real SandboxPolicy proto needs the platform-restricted 'openshell' extra;
# skip those cases where it isn't installed (e.g. CI). The dict-only helpers below
# (generate_policy_dict / inject_platform_egress) run everywhere.
requires_openshell = pytest.mark.skipif(
    importlib.util.find_spec("openshell") is None,
    reason="openshell extra not installed (platform-restricted wheel)",
)

_FULL = {
    "version": 1,
    "filesystem_policy": {
        "include_workdir": True,
        "read_only": ["/usr", "/opt"],
        "read_write": ["/home/sandbox", "/dev/shm"],
    },
    "landlock": {"compatibility": "best_effort"},
    "process": {"run_as_user": "sandbox", "run_as_group": "sandbox"},
    "network_policies": {
        "igw": {
            "name": "nemo-igw",
            "endpoints": [
                {
                    "host": "host.docker.internal",
                    "port": 8080,
                    "protocol": "rest",
                    "enforcement": "enforce",
                    "access": "full",
                }
            ],
            "binaries": [{"path": "/usr/bin/curl"}, {"path": "/workspace/.venv/bin/python3.13"}],
        }
    },
}


@requires_openshell
def test_build_sandbox_policy_full() -> None:
    policy = build_sandbox_policy(_FULL)

    assert policy.version == 1
    assert policy.filesystem.include_workdir is True
    assert list(policy.filesystem.read_only) == ["/usr", "/opt"]
    assert "/dev/shm" in policy.filesystem.read_write
    assert policy.landlock.compatibility == "best_effort"
    assert policy.process.run_as_user == "sandbox"
    assert policy.process.run_as_group == "sandbox"

    assert "igw" in policy.network_policies
    rule = policy.network_policies["igw"]
    assert rule.name == "nemo-igw"
    assert rule.endpoints[0].host == "host.docker.internal"
    assert rule.endpoints[0].port == 8080
    assert rule.endpoints[0].access == "full"
    assert [b.path for b in rule.binaries] == ["/usr/bin/curl", "/workspace/.venv/bin/python3.13"]


@requires_openshell
def test_build_sandbox_policy_minimal() -> None:
    policy = build_sandbox_policy({})
    assert policy.version == 1
    assert not policy.filesystem.read_only
    assert not policy.network_policies


@requires_openshell
def test_build_sandbox_policy_accepts_filesystem_alias() -> None:
    # proto field name 'filesystem' is accepted as well as the YAML 'filesystem_policy'
    policy = build_sandbox_policy({"filesystem": {"read_write": ["/tmp"]}})
    assert list(policy.filesystem.read_write) == ["/tmp"]


def _egress() -> HelixEgress:
    return HelixEgress(host="host.docker.internal", port=8080, binaries=("/workspace/.venv/bin/python3.13",))


def test_generate_policy_dict_is_default_deny() -> None:
    policy = generate_policy_dict(filesystem=SandboxFilesystem(), egress=_egress())

    # exactly one network rule: the platform egress. everything else denied.
    assert list(policy["network_policies"]) == [PLATFORM_EGRESS_KEY]
    rule = policy["network_policies"][PLATFORM_EGRESS_KEY]
    assert rule["endpoints"][0] == {
        "host": "host.docker.internal",
        "port": 8080,
        "protocol": "rest",
        "enforcement": "enforce",
        "access": "full",
    }
    assert [b["path"] for b in rule["binaries"]] == ["/workspace/.venv/bin/python3.13"]
    # sandbox exec/write paths + run-as come from the filesystem shape.
    assert "/opt" in policy["filesystem_policy"]["read_only"]
    assert "/workspace" in policy["filesystem_policy"]["read_write"]
    assert "/home/sandbox" in policy["filesystem_policy"]["read_write"]
    assert "/dev/shm" in policy["filesystem_policy"]["read_write"]
    assert policy["process"]["run_as_user"] == "sandbox"


def test_generate_policy_dict_no_egress_is_pure_default_deny() -> None:
    # egress=None (gateway-managed inference.local) -> no network rules at all.
    policy = generate_policy_dict(filesystem=SandboxFilesystem(), egress=None)
    assert policy["network_policies"] == {}
    # filesystem/process shape still applied.
    assert "/opt" in policy["filesystem_policy"]["read_only"]
    assert policy["process"]["run_as_user"] == "sandbox"


@requires_openshell
def test_generate_sandbox_policy_no_egress() -> None:
    policy = generate_sandbox_policy(filesystem=SandboxFilesystem(), egress=None)
    assert not policy.network_policies


@requires_openshell
def test_generate_sandbox_policy_builds_proto() -> None:
    policy = generate_sandbox_policy(filesystem=SandboxFilesystem(), egress=_egress())
    assert PLATFORM_EGRESS_KEY in policy.network_policies
    rule = policy.network_policies[PLATFORM_EGRESS_KEY]
    assert rule.endpoints[0].host == "host.docker.internal"
    assert rule.endpoints[0].port == 8080
    assert [b.path for b in rule.binaries] == ["/workspace/.venv/bin/python3.13"]
    assert policy.process.run_as_user == "sandbox"


def test_inject_platform_egress_overwrites_user_rule_and_keeps_others() -> None:
    # a user policy that tries to point the platform key elsewhere, plus an unrelated rule.
    policy = {
        "network_policies": {
            PLATFORM_EGRESS_KEY: {"name": "evil", "endpoints": [{"host": "attacker.example", "port": 443}]},
            "pypi": {"name": "pypi", "endpoints": [{"host": "pypi.org", "port": 443}]},
        }
    }
    inject_platform_egress(policy, _egress())

    rule = policy["network_policies"][PLATFORM_EGRESS_KEY]
    assert rule["endpoints"][0]["host"] == "host.docker.internal"  # ours, not the user's
    assert "pypi" in policy["network_policies"]  # unrelated rule preserved


def test_inject_platform_egress_adds_rule_when_absent() -> None:
    policy: dict = {"version": 1}
    inject_platform_egress(policy, _egress())
    assert PLATFORM_EGRESS_KEY in policy["network_policies"]


def test_platform_egress_tls_omitted_when_empty() -> None:
    plain = generate_policy_dict(filesystem=SandboxFilesystem(), egress=_egress())
    assert "tls" not in plain["network_policies"][PLATFORM_EGRESS_KEY]["endpoints"][0]

    secure = generate_policy_dict(
        filesystem=SandboxFilesystem(),
        egress=HelixEgress(host="h", port=443, tls="terminate"),
    )
    assert secure["network_policies"][PLATFORM_EGRESS_KEY]["endpoints"][0]["tls"] == "terminate"


@requires_openshell
def test_load_sandbox_policy_reads_yaml(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(
        "version: 1\nprocess:\n  run_as_user: sandbox\n  run_as_group: sandbox\n",
        encoding="utf-8",
    )
    policy = load_sandbox_policy(str(policy_file))
    assert policy.process.run_as_user == "sandbox"


# --- finding #5: fail-open on loaded-policy + Landlock paths ---


@requires_openshell
def test_network_rule_defaults_enforcement_to_enforce() -> None:
    # a loaded/override rule that omits `enforcement` must block (proto default is audit).
    policy = build_sandbox_policy({"network_policies": {"r": {"endpoints": [{"host": "h", "port": 8080}]}}})
    assert policy.network_policies["r"].endpoints[0].enforcement == "enforce"


@requires_openshell
def test_build_sandbox_policy_full_keeps_explicit_enforcement() -> None:
    # explicit enforcement is preserved (regression guard for the new default).
    policy = build_sandbox_policy(_FULL)
    assert policy.network_policies["igw"].endpoints[0].enforcement == "enforce"


def test_normalize_loaded_policy_injects_sandbox_process_when_absent() -> None:
    data = normalize_loaded_policy({"version": 1})
    assert data["process"]["run_as_user"] == "sandbox"
    assert data["process"]["run_as_group"] == "sandbox"


def test_normalize_loaded_policy_injects_landlock_when_absent() -> None:
    data = normalize_loaded_policy({"version": 1})
    assert data["landlock"]["compatibility"] == "best_effort"


def test_normalize_loaded_policy_preserves_authored_blocks() -> None:
    data = normalize_loaded_policy(
        {
            "process": {"run_as_user": "custom", "run_as_group": "custom"},
            "landlock": {"compatibility": "hard_requirement"},
        }
    )
    assert data["process"]["run_as_user"] == "custom"
    assert data["landlock"]["compatibility"] == "hard_requirement"


def test_normalize_loaded_policy_honors_landlock_override() -> None:
    data = normalize_loaded_policy({"version": 1}, landlock_compatibility="hard_requirement")
    assert data["landlock"]["compatibility"] == "hard_requirement"


def test_normalize_loaded_policy_fills_partial_process() -> None:
    data = normalize_loaded_policy({"process": {"run_as_group": "grp"}})
    assert data["process"]["run_as_user"] == "sandbox"  # missing key filled
    assert data["process"]["run_as_group"] == "grp"  # authored key kept


@requires_openshell
def test_load_sandbox_policy_injects_defaults_when_blocks_absent(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("version: 1\n", encoding="utf-8")
    policy = load_sandbox_policy(str(policy_file))
    assert policy.process.run_as_user == "sandbox"
    assert policy.process.run_as_group == "sandbox"
    assert policy.landlock.compatibility == "best_effort"


def test_generate_policy_dict_honors_landlock_compatibility() -> None:
    # the knob threads through SandboxFilesystem into the generated policy dict.
    policy = generate_policy_dict(filesystem=SandboxFilesystem(landlock_compatibility="hard_requirement"), egress=None)
    assert policy["landlock"]["compatibility"] == "hard_requirement"


def test_executor_config_landlock_default_and_validation() -> None:
    from nemo_deployments_plugin.backends.openshell.config import OpenShellExecutorConfig

    assert OpenShellExecutorConfig().landlock_compatibility == "best_effort"
    assert (
        OpenShellExecutorConfig(landlock_compatibility="hard_requirement").landlock_compatibility == "hard_requirement"
    )
    with pytest.raises(ValueError):
        OpenShellExecutorConfig(landlock_compatibility="bogus")


# --- policy overrides are validated strictly ---


@requires_openshell
def test_build_sandbox_policy_rejects_landlock_key_typo() -> None:
    # A typo'd key must not leave the sandbox on the fail-open landlock default.
    with pytest.raises(ValueError, match="compatibilty"):
        build_sandbox_policy({"landlock": {"compatibilty": "hard_requirement"}})


@requires_openshell
def test_build_sandbox_policy_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ValueError, match="filesystem_polcy"):
        build_sandbox_policy({"version": 1, "filesystem_polcy": {"read_only": ["/usr"]}})


@requires_openshell
def test_build_sandbox_policy_rejects_bad_landlock_value() -> None:
    # The proto types this as a plain string and the supervisor reads anything it does not
    # recognise as best-effort, so the value set is checked here.
    with pytest.raises(ValueError, match="landlock.compatibility"):
        build_sandbox_policy({"landlock": {"compatibility": "hard-requirement"}})


@requires_openshell
def test_build_sandbox_policy_rejects_bad_enforcement_value() -> None:
    with pytest.raises(ValueError, match="enforcement"):
        build_sandbox_policy({"network_policies": {"r": {"endpoints": [{"host": "h", "enforcement": "off"}]}}})


@requires_openshell
def test_build_sandbox_policy_accepts_field_this_backend_never_writes() -> None:
    # Structural validation comes from the proto, so a valid OpenShell policy this backend
    # does not generate still round-trips.
    policy = build_sandbox_policy(
        {"network_policies": {"r": {"endpoints": [{"host": "h", "allowed_ips": ["10.0.0.1"], "ports": [443]}]}}}
    )
    assert list(policy.network_policies["r"].endpoints[0].allowed_ips) == ["10.0.0.1"]


@requires_openshell
def test_build_sandbox_policy_accepts_both_filesystem_keys() -> None:
    by_alias = build_sandbox_policy({"filesystem_policy": {"read_only": ["/usr"]}})
    by_name = build_sandbox_policy({"filesystem": {"read_only": ["/usr"]}})
    assert by_alias == by_name


@requires_openshell
def test_load_sandbox_policy_rejects_typo_in_file(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("version: 1\nlandlock:\n  compatibilty: hard_requirement\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid sandbox policy"):
        load_sandbox_policy(str(policy_file))
