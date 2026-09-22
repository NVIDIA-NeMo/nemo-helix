# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and generate an OpenShell ``SandboxPolicy`` proto.

Two producers, one proto:

- ``load_sandbox_policy`` reads a hand-written policy YAML.
- ``generate_sandbox_policy`` builds a default-deny policy structurally from a
  sandbox's own filesystem shape + the platform egress it must reach.

Both go through ``build_sandbox_policy`` so the mapping from the documented YAML
shape (``version`` / ``filesystem_policy`` / ``landlock`` / ``process`` /
``network_policies``) to the proto lives in one place. Everything here is a
property of the *sandbox*, not of a deployment: it takes plain structured inputs
and returns a policy, so a future non-deployment consumer (a sandboxed job, a
tool runner) can reuse it without importing deployment types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import yaml
from google.protobuf import json_format
from nemo_deployments_plugin.backends.base import MissingBackendDependencyError

if TYPE_CHECKING:
    from openshell._proto import sandbox_pb2 as sb  # ty: ignore[unresolved-import]

# Default filesystem/process shape for a NAT-agent OpenShell image, verified against
# a live gateway: /opt holds the uv-managed interpreter the venv nat
# launcher execs; /workspace is the agent venv; /home/sandbox is the serve workdir
# (NAT writes <cwd>/.tmp); /dev/shm is needed by nat[most]'s Dask cluster (POSIX
# semaphores). These describe the image, so they belong with the sandbox, not the
# deployment.
DEFAULT_READ_ONLY: tuple[str, ...] = ("/usr", "/bin", "/lib", "/etc", "/opt", "/proc", "/dev/urandom", "/var/log")
DEFAULT_READ_WRITE: tuple[str, ...] = ("/workspace", "/home/sandbox", "/sandbox", "/tmp", "/dev/null", "/dev/shm")
DEFAULT_RUN_AS_USER = "sandbox"

# Landlock compatibility default. "best_effort" runs on kernels without Landlock but
# degrades to NO filesystem confinement there (fail-open); "hard_requirement" fails
# the sandbox closed on such kernels. Kept as the default to preserve the local-dev
# docker-driver happy path; harden via OpenShellExecutorConfig.landlock_compatibility.
DEFAULT_LANDLOCK_COMPATIBILITY: Literal["best_effort", "hard_requirement"] = "best_effort"

# The binary that opens the egress socket for NAT's LLM calls is the venv/uv python,
# NOT curl. The exact uv-managed interpreter path is image/patch-version specific
# (e.g. /opt/uv/python/cpython-3.13.7-.../bin/python3.13); an executor can add it via
# its platform_egress config. OpenShell matches binary paths exactly, with no prefix or
# glob support, so an egress rule has to pin the interpreter patch version.
DEFAULT_EGRESS_BINARIES: tuple[str, ...] = ("/workspace/.venv/bin/python3.13", "/usr/bin/curl")

# Map key for the mandatory platform egress rule. Reserved: injected into every
# policy, so a user rule at this key is overwritten rather than merged.
PLATFORM_EGRESS_KEY = "nemo_helix"


# Values the supervisor recognises for the two free-form string fields that fail OPEN when
# unrecognised: anything but "hard_requirement" becomes best-effort Landlock (no filesystem
# confinement on a kernel without Landlock), and an unset enforcement takes the proto's
# "audit" default instead of blocking. The proto cannot type these, so they are checked here.
LANDLOCK_COMPATIBILITIES = ("best_effort", "hard_requirement")
ENFORCEMENTS = ("enforce", "audit")
DEFAULT_ENFORCEMENT = "enforce"

# OpenShell's baseline policy is version 1; an omitted version would parse as the proto's 0.
DEFAULT_POLICY_VERSION = 1


@dataclass(frozen=True)
class SandboxFilesystem:
    """Filesystem + process shape a sandbox image needs to boot and run the agent."""

    read_only: tuple[str, ...] = DEFAULT_READ_ONLY
    read_write: tuple[str, ...] = DEFAULT_READ_WRITE
    include_workdir: bool = True
    run_as_user: str = DEFAULT_RUN_AS_USER
    run_as_group: str = DEFAULT_RUN_AS_USER
    landlock_compatibility: str = DEFAULT_LANDLOCK_COMPATIBILITY


@dataclass(frozen=True)
class HelixEgress:
    """The one egress a sandbox must always be allowed: the NeMo Helix (Inference Gateway/entities/files).

    Environment-specific (docker driver -> host.docker.internal:8080; k8s -> the
    platform Service). This is the sole allowed rule in a generated default-deny
    policy, and is re-injected into any static/overridden policy so a deployment can
    never lose its path home.
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
    root); one that omits ``landlock`` would ship no filesystem confinement. The
    generated path always sets both, so mirror it here: default the process block to
    the sandbox user/group and default the landlock block. Blocks the author wrote are
    kept as-is. Mutates and returns ``data``.
    """
    process = data.get("process")
    if not isinstance(process, dict):
        process = {}
        data["process"] = process
    process.setdefault("run_as_user", DEFAULT_RUN_AS_USER)
    process.setdefault("run_as_group", DEFAULT_RUN_AS_USER)

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
    rule. When ``None`` (e.g. gateway-managed inference via ``inference.local``,
    which needs no sandbox egress), the policy allows no egress at all, a pure
    default-deny policy.
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

    Applied to every policy (generated, static YAML, or a future user override) so a
    policy can never sever the sandbox's path back to the platform. Mutates and
    returns ``policy``.
    """
    network = policy.setdefault("network_policies", {})
    if not isinstance(network, dict):
        raise ValueError("network_policies must be a mapping of rule name -> rule")
    network[egress.key] = _platform_egress_rule(egress)
    return policy


def generate_sandbox_policy(*, filesystem: SandboxFilesystem, egress: HelixEgress | None = None) -> Any:
    """Generate a default-deny ``SandboxPolicy`` proto from structured inputs.

    ``egress=None`` yields a policy with no egress at all (gateway-managed inference).
    """
    return build_sandbox_policy(generate_policy_dict(filesystem=filesystem, egress=egress))


_OPENSHELL_INSTALL_HINT = (
    "The 'openshell' package is required to build OpenShell sandbox policies. "
    'Install it with: uv pip install "openshell>=0.0.92" "grpcio>=1.78.0" "protobuf>=6.31.1"'
)


def _ensure_sb() -> None:
    """Import the openshell sandbox proto lazily and bind it as a module global, so
    this module loads without the optional ``openshell`` package (matching the
    docker/k8s backends). The first policy build triggers this.
    """
    if globals().get("sb") is not None:
        return
    try:
        from openshell._proto import sandbox_pb2 as sb  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise MissingBackendDependencyError(_OPENSHELL_INSTALL_HINT) from exc
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


def _apply_defaults_and_check(policy: Any) -> None:
    """Fill the defaults the proto cannot express, and check the fields it cannot type.

    The supervisor reads ``compatibility`` and ``enforcement`` as free-form strings and
    treats anything it does not recognise as the weaker setting, so an unrecognised value
    here would ship a policy that reads as confined but is not.
    """
    if not policy.version:
        policy.version = DEFAULT_POLICY_VERSION
    if policy.HasField("landlock") and policy.landlock.compatibility not in LANDLOCK_COMPATIBILITIES:
        raise ValueError(
            f"invalid sandbox policy: landlock.compatibility must be one of "
            f"{', '.join(LANDLOCK_COMPATIBILITIES)}, got '{policy.landlock.compatibility}'"
        )
    for key, rule in policy.network_policies.items():
        for endpoint in rule.endpoints:
            if not endpoint.enforcement:
                endpoint.enforcement = DEFAULT_ENFORCEMENT
            elif endpoint.enforcement not in ENFORCEMENTS:
                raise ValueError(
                    f"invalid sandbox policy: network_policies.{key} enforcement must be one of "
                    f"{', '.join(ENFORCEMENTS)}, got '{endpoint.enforcement}'"
                )


def build_sandbox_policy(data: dict[str, Any]) -> Any:
    """Parse a policy mapping into a ``SandboxPolicy`` proto, rejecting anything unknown.

    The proto is the policy schema, so it does the structural validation: a misspelled or
    unknown key raises here instead of being dropped on the way to a weaker policy, and
    every field OpenShell supports is accepted, including ones this backend never writes.
    """
    _ensure_sb()
    try:
        policy = json_format.ParseDict(_with_proto_field_names(data), sb.SandboxPolicy())
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
