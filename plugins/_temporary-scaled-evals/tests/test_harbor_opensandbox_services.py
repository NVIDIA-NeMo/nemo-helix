# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation of an evaluation profile's ``compose_services``."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("scaled_evals", reason="scaled-evals plugin not installed")

from pydantic import ValidationError
from scaled_evals.harbor_opensandbox_services import ComposeServices, parse_compose_services


def _spec(*services: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """A raw ``compose_services`` mapping: these services (one Postgres by default), plus top-level ``overrides``."""
    return {"services": list(services) or [{"name": "db", "image": "docker.io/library/postgres:16"}], **overrides}


def test_round_trips_through_the_extension_value() -> None:
    spec = parse_compose_services(
        _spec(
            {"name": "db", "image": "docker.io/library/postgres:16", "readiness": {"exec": ["pg_isready"]}},
            {"name": "seed", "image": "localhost:5000/seed:1", "role": "run_once"},
            volumes=["shared"],
            main={"volume_mounts": [{"name": "shared", "mount_path": "/shared"}], "add_capabilities": ["SYS_PTRACE"]},
        )
    )

    assert ComposeServices.model_validate_json(spec.to_json()) == spec
    assert '"exec":["pg_isready"]' in spec.to_json()
    assert spec.names() == ["db", "seed"]
    assert [s.role for s in spec.services] == ["sidecar", "run_once"]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_spec({"name": "main", "image": "docker.io/x:1"}), "service name 'main' is reserved"),
        (_spec({"name": "egress", "image": "docker.io/x:1"}), "service name 'egress' is reserved"),
        (_spec({"name": "odoo", "image": "docker.io/x:1", "ports": [18080]}), r"ports \[18080\] are used"),
        (_spec({"name": "db", "image": "postgres:16"}), "String should match pattern"),
        (
            _spec({"name": "seed", "image": "docker.io/x:1", "role": "run_once", "readiness": {"exec": ["true"]}}),
            "drop readiness",
        ),
        (
            _spec(
                {"name": "a", "image": "docker.io/x:1", "ports": [80]},
                {"name": "b", "image": "docker.io/y:1", "ports": [80]},
            ),
            "same port",
        ),
        (_spec({"name": "a", "image": "docker.io/x:1"}, {"name": "a", "image": "docker.io/y:1"}), "unique"),
        (_spec(main={"volume_mounts": [{"name": "data", "mount_path": "/data"}]}), "'data' is not declared"),
        (_spec({"name": "a", "image": "docker.io/x:1", "role": "sidekick"}), "Input should be"),
        (_spec({"name": "a", "image": "docker.io/x:1", "privileged": True}), "Extra inputs are not permitted"),
        (_spec({"name": "a", "image": "docker.io/x:1", "ports": ["80"]}), "valid integer"),
        (_spec(main={"add_capabilities": ["NET_ADMIN"]}), "Input should be 'SYS_PTRACE'"),
    ],
)
def test_rejects_specs_the_pod_cant_run(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        parse_compose_services(raw)


def test_rejects_a_non_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_compose_services(["db"])


def test_sandbox_entrypoint_keeps_main_alive() -> None:
    assert parse_compose_services(_spec()).sandbox_entrypoint() is None
    assert parse_compose_services(_spec(main={"entrypoint": ["/start.sh"]})).sandbox_entrypoint() == [
        "/start.sh",
        "sh",
        "-c",
        "sleep infinity",
    ]
