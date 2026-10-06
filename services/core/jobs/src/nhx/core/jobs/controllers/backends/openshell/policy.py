# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an OpenShell ``SandboxPolicy`` proto for job sandboxes.

This is a separate copy from the deployments plugin's policy builder, so the
jobs service does not depend on that plugin. It mirrors that builder with
job-specific defaults: the task venv
and scratch storage shape of the nhx-tasks-openshell image, and an egress
allowlist restricted to the jobs-launcher and the task interpreter.

Everything here is a property of the *sandbox*, not of a job: plain structured
inputs in, a policy proto out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import yaml
from google.protobuf import json_format
from nemo_helix_plugin.capabilities import CapabilityUnavailableError

if TYPE_CHECKING:
    from openshell._proto import sandbox_pb2 as sb  # ty: ignore[unresolved-import]

# Filesystem shape of the nhx-tasks-openshell image (Dockerfile.nhx-tasks):
# /app holds the read-only venv, /tools the baked jobs-launcher, /var/run/scratch
# the task/job/config storage paths the jobs controller sets, /home/sandbox the
# sandbox identity's home. /dev/shm is the launcher's marker tmpfs.
DEFAULT_READ_ONLY: tuple[str, ...] = (
    "/usr",
    "/bin",
    "/lib",
    "/etc",
    "/opt",
    "/proc",
    "/dev/urandom",
    "/var/log",
    "/app",
    "/tools",
)
DEFAULT_READ_WRITE: tuple[str, ...] = ("/var/run/scratch", "/home/sandbox", "/tmp", "/dev/null", "/dev/shm")
DEFAULT_RUN_AS_USER = "sandbox"
# The openshell image variant creates this user (values mirror the deployments
# sandbox profile).
DEFAULT_RUN_AS_GROUP = "sandbox"

# Landlock compatibility default. "best_effort" runs on kernels without Landlock but
# degrades to NO filesystem confinement there (fail-open); "hard_requirement" fails
# the sandbox closed on such kernels. Kept as the default to preserve the local-dev
# docker-driver happy path; harden via the profile config.
DEFAULT_LANDLOCK_COMPATIBILITY: Literal["best_effort", "hard_requirement"] = "best_effort"

# The only binaries allowed to open the platform egress connection: the
# jobs-launcher (step-config and secrets fetch) and the task interpreter that
# talks to the platform (files, entities, ...). OpenShell matches the resolved
# executable, so the venv symlink (/app/.venv/bin/python -> the base image
# interpreter) never matches; the real path is the base image's interpreter.
# Binary paths are globs where `*` stays within one path component.
DEFAULT_EGRESS_BINARIES: tuple[str, ...] = ("/tools/jobs-launcher", "/usr/local/bin/python3*")

# Map key for the mandatory platform egress rule. Reserved: injected into every
# policy, so a user rule at this key is overwritten rather than merged.
PLATFORM_EGRESS_KEY = "nemo_helix"

LANDLOCK_COMPATIBILITIES = ("best_effort", "hard_requirement")

# Policy YAML spells these endpoint fields as strings; the proto types them as enums.
_ENDPOINT_ENUMS: dict[str, tuple[str, dict[str, str]]] = {
    "enforcement": (
        "NETWORK_ENFORCEMENT_MODE",
        {"": "UNSPECIFIED", "enforce": "ENFORCE", "audit": "AUDIT"},
    ),
    "access": (
        "NETWORK_ACCESS_PRESET",
        {"": "UNSPECIFIED", "read-only": "READ_ONLY", "read-write": "READ_WRITE", "full": "FULL"},
    ),
    "tls": (
        "NETWORK_TLS_MODE",
        {"": "UNSPECIFIED", "skip": "SKIP"},
    ),
}
DEFAULT_ENFORCEMENT = "enforce"
DEFAULT_POLICY_VERSION = 1

_OPENSHELL_INSTALL_HINT = (
    "The 'openshell' package is required to build OpenShell sandbox policies. "
    'Install it with: uv pip install "openshell>=0.1.2" "grpcio>=1.78.0" "protobuf>=6.31.1"'
)


@dataclass(frozen=True)
class SandboxFilesystem:
    """Filesystem + process shape a job sandbox image needs to boot and run."""

    read_only: tuple[str, ...] = DEFAULT_READ_ONLY
    read_write: tuple[str, ...] = DEFAULT_READ_WRITE
    include_workdir: bool = True
    run_as_user: str = DEFAULT_RUN_AS_USER
    run_as_group: str = DEFAULT_RUN_AS_GROUP
    landlock_compatibility: str = DEFAULT_LANDLOCK_COMPATIBILITY


@dataclass(frozen=True)
class HelixEgress:
    """The one egress a job sandbox must always be allowed: the NeMo Helix platform.

    Environment-specific (docker driver -> host.docker.internal:8080; k8s -> the
    platform Service). This is the sole allowed rule in a generated default-deny
    policy, and is re-injected into any static policy so a job can never lose its
    path home.
    """

    host: str
    port: int
    protocol: str = "rest"
    tls: str = ""
    access: str = "full"
    enforcement: str = "enforce"
    binaries: tuple[str, ...] = DEFAULT_EGRESS_BINARIES
    name: str = "nemo-helix-egress"
    key: str = PLATFORM_EGRESS_KEY


def load_policy_dict(path: str) -> dict[str, Any]:
    """Read a policy YAML file and return the parsed mapping (not yet a proto)."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"sandbox policy file {path} must contain a YAML mapping")
    return data


def normalize_loaded_policy(
    data: dict[str, Any], *, landlock_compatibility: str = DEFAULT_LANDLOCK_COMPATIBILITY
) -> dict[str, Any]:
    """Inject safe defaults into a hand-written policy mapping before it becomes a proto.

    A loaded YAML that omits ``process`` would run as the image-default user (possibly
    root); one that omits ``landlock`` would ship no filesystem confinement. Default the
    process block to the sandbox user/group and default the landlock block; blocks the
    author wrote are kept as-is. Mutates and returns ``data``.
    """
    process = data.get("process")
    if not isinstance(process, dict):
        process = {}
        data["process"] = process
    process.setdefault("run_as_user", DEFAULT_RUN_AS_USER)
    process.setdefault("run_as_group", DEFAULT_RUN_AS_GROUP)

    landlock = data.get("landlock")
    if not isinstance(landlock, dict):
        data["landlock"] = {"compatibility": landlock_compatibility}
    else:
        landlock.setdefault("compatibility", landlock_compatibility)
    return data


def load_sandbox_policy(path: str) -> Any:
    """Read a policy YAML file and return a ``SandboxPolicy`` proto."""
    return build_sandbox_policy(normalize_loaded_policy(load_policy_dict(path)))


def generate_policy_dict(*, filesystem: SandboxFilesystem, egress: HelixEgress | None) -> dict[str, Any]:
    """Generate a default-deny policy mapping.

    When ``egress`` is given, the platform egress rule is the sole allowed network
    rule. When ``None``, the policy allows no egress at all, a pure default-deny
    policy (breaks secrets fetch and OTLP log export; the profile default is to set
    platform_egress).
    """
    return {
        "version": 1,
        "filesystem_policy": {
            "include_workdir": filesystem.include_workdir,
            "read_only": list(filesystem.read_only),
            "read_write": list(filesystem.read_write),
        },
        "landlock": {"compatibility": filesystem.landlock_compatibility},
        "process": {"run_as_user": filesystem.run_as_user, "run_as_group": filesystem.run_as_group},
        "network_policies": ({egress.key: _platform_egress_rule(egress)} if egress is not None else {}),
    }


def inject_platform_egress(policy: dict[str, Any], egress: HelixEgress) -> dict[str, Any]:
    """Ensure the mandatory platform egress rule is present, overwriting any rule at its key.

    Applied to every policy (generated or static YAML) so a policy can never sever the
    sandbox's path back to the platform. Mutates and returns ``policy``.
    """
    network = policy.setdefault("network_policies", {})
    if not isinstance(network, dict):
        raise ValueError("network_policies must be a mapping of rule name -> rule")
    network[egress.key] = _platform_egress_rule(egress)
    return policy


def generate_sandbox_policy(*, filesystem: SandboxFilesystem, egress: HelixEgress | None = None) -> Any:
    """Generate a default-deny ``SandboxPolicy`` proto from structured inputs."""
    return build_sandbox_policy(generate_policy_dict(filesystem=filesystem, egress=egress))


def _ensure_sb() -> None:
    """Import the openshell sandbox proto lazily, so this module loads without the
    optional ``openshell`` package. The first policy build triggers this.
    """
    if globals().get("sb") is not None:
        return
    try:
        from openshell._proto import sandbox_pb2 as sb  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise CapabilityUnavailableError(_OPENSHELL_INSTALL_HINT) from exc
    globals()["sb"] = sb


def _with_proto_field_names(data: dict[str, Any]) -> dict[str, Any]:
    """Rename the documented ``filesystem_policy`` key to the proto's ``filesystem``."""
    if "filesystem_policy" not in data:
        return data
    renamed = dict(data)
    filesystem = renamed.pop("filesystem_policy")
    if "filesystem" in renamed and renamed["filesystem"] != filesystem:
        raise ValueError("sandbox policy sets both filesystem_policy and filesystem; keep one")
    renamed["filesystem"] = filesystem
    return renamed


def _with_proto_enum_values(data: dict[str, Any]) -> dict[str, Any]:
    """Translate endpoint ``enforcement``/``access``/``tls`` YAML strings to proto enum names.

    An unset or empty enforcement becomes ``enforce``: the supervisor treats an
    unspecified enforcement as audit, which would ship a rule that reads as blocking but
    only logs. Unknown strings raise rather than falling through to a weaker setting.
    """
    policies = data.get("network_policies")
    if not isinstance(policies, dict):
        return data
    translated_policies: dict[str, Any] = {}
    for key, rule in policies.items():
        endpoints = rule.get("endpoints") if isinstance(rule, dict) else None
        if not isinstance(endpoints, list):
            translated_policies[key] = rule
            continue
        translated_endpoints = []
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                translated_endpoints.append(endpoint)
                continue
            endpoint = dict(endpoint)
            endpoint["enforcement"] = endpoint.get("enforcement") or DEFAULT_ENFORCEMENT
            for field, (prefix, values) in _ENDPOINT_ENUMS.items():
                if field not in endpoint:
                    continue
                value = endpoint[field]
                if value not in values:
                    raise ValueError(
                        f"invalid sandbox policy: network_policies.{key} {field} must be one of "
                        f"{', '.join(repr(v) for v in values)}, got {value!r}"
                    )
                endpoint[field] = f"{prefix}_{values[value]}"
            translated_endpoints.append(endpoint)
        translated_policies[key] = {**rule, "endpoints": translated_endpoints}
    return {**data, "network_policies": translated_policies}


def _apply_defaults_and_check(policy: Any) -> None:
    """Fill the defaults the proto cannot express, and check the fields it cannot type.

    The supervisor reads ``compatibility`` as a free-form string and treats anything it
    does not recognise as the weaker setting, so an unrecognised value here would ship a
    policy that reads as confined but is not.
    """
    if not policy.version:
        policy.version = DEFAULT_POLICY_VERSION
    if policy.HasField("landlock") and policy.landlock.compatibility not in LANDLOCK_COMPATIBILITIES:
        raise ValueError(
            f"invalid sandbox policy: landlock.compatibility must be one of "
            f"{', '.join(LANDLOCK_COMPATIBILITIES)}, got '{policy.landlock.compatibility}'"
        )


def build_sandbox_policy(data: dict[str, Any]) -> Any:
    """Parse a policy mapping into a ``SandboxPolicy`` proto, rejecting anything unknown.

    The proto is the policy schema, so it does the structural validation: a misspelled or
    unknown key raises here instead of being dropped on the way to a weaker policy.
    """
    _ensure_sb()
    try:
        policy = json_format.ParseDict(_with_proto_enum_values(_with_proto_field_names(data)), sb.SandboxPolicy())
    except json_format.ParseError as exc:
        raise ValueError(f"invalid sandbox policy: {exc}") from exc
    _apply_defaults_and_check(policy)
    return policy


def _platform_egress_rule(egress: HelixEgress) -> dict[str, Any]:
    """The platform egress rule in the policy-YAML dict shape."""
    endpoint: dict[str, Any] = {
        "host": egress.host,
        "port": egress.port,
        "protocol": egress.protocol,
        "enforcement": egress.enforcement,
        "access": egress.access,
    }
    if egress.tls:
        endpoint["tls"] = egress.tls
    return {
        "name": egress.name,
        "endpoints": [endpoint],
        "binaries": [{"path": p} for p in egress.binaries],
    }
