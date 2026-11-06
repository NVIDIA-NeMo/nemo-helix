# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_helix_plugin.agent_hardener.types import (
    ManifestBackend,
    ManifestInit,
    ManifestUpdate,
    SynthBenignSpec,
    WarGameSpec,
)
from pydantic import ValidationError

SPEC_TYPES = [
    pytest.param(lambda **kw: WarGameSpec(manifest_id="m1", **kw), id="war-game"),
    pytest.param(lambda **kw: SynthBenignSpec(manifest_id="m1", **kw), id="synth"),
]


@pytest.mark.parametrize("make_spec", SPEC_TYPES)
@pytest.mark.parametrize("ref", ["registry.example.com/agent:1", "agent@sha256:" + "a" * 64, "localhost:5000/a/b"])
def test_specs_accept_an_image_reference(make_spec, ref: str) -> None:
    spec = make_spec(victim_image=ref, allow_context_drift=True)

    assert spec.victim_image == ref
    assert spec.allow_context_drift is True


@pytest.mark.parametrize("make_spec", SPEC_TYPES)
@pytest.mark.parametrize("ref", ["", " ", "agent image:1", "--allow-context-drift", "-x", "agent:1\n"])
def test_specs_reject_a_value_that_is_not_one_image_reference(make_spec, ref: str) -> None:
    with pytest.raises(ValidationError, match="victim_image"):
        make_spec(victim_image=ref)


@pytest.mark.parametrize("make_spec", SPEC_TYPES)
def test_specs_allow_drift_for_the_saved_manifests_image(make_spec) -> None:
    assert make_spec(allow_context_drift=True).allow_context_drift is True


def test_a_local_config_rejects_allow_context_drift_without_an_image() -> None:
    with pytest.raises(ValidationError, match="allow_context_drift requires victim_image"):
        WarGameSpec(config="agent-hardener.yaml", allow_context_drift=True)


@pytest.mark.parametrize("make_spec", SPEC_TYPES)
def test_specs_default_to_a_local_build(make_spec) -> None:
    spec = make_spec()

    assert spec.victim_image is None
    assert spec.allow_context_drift is False


def test_new_spec_fields_keep_the_legacy_job_kind_keys_apart() -> None:
    war_game = WarGameSpec(manifest_id="m1", victim_image="agent:1").model_dump(mode="json")
    synth = SynthBenignSpec(manifest_id="m1", victim_image="agent:1").model_dump(mode="json")

    assert "stop_after_synth" in war_game and "interview" not in war_game
    assert "interview" in synth and "stop_after_synth" not in synth


@pytest.mark.parametrize("host", ["ledger.ns.svc", "ledger", "ledger.ns.svc.cluster.local.", "10.96.0.12", "fd00::12"])
def test_backend_accepts_a_hostname_or_ip(host: str) -> None:
    assert ManifestBackend(name="ledger", ports=[8087], host=host).host == host


@pytest.mark.parametrize(
    "host", ["http://ledger", "ledger:8087", "ledger/x", "[fd00::12]", "led ger", "-ledger", "ledger_svc", "a" * 64]
)
def test_backend_rejects_a_host_that_is_not_bare(host: str) -> None:
    with pytest.raises(ValidationError, match="bare hostname"):
        ManifestBackend(name="ledger", ports=[8087], host=host)


def test_backend_treats_a_blank_host_as_unset() -> None:
    assert ManifestBackend(name="ledger", ports=[8087], host="  ").host is None


@pytest.mark.parametrize("ips", [["10.96.0.0/12"], ["10.96.0.12"], ["fd00::/8", "fd00::12"]])
def test_backend_accepts_ips_and_cidrs(ips: list[str]) -> None:
    assert ManifestBackend(name="ledger", ports=[8087], allowed_ips=ips).allowed_ips == ips


@pytest.mark.parametrize("ip", ["10.96.0.1/12", "ledger.ns.svc", "10.96.0.0/33", ""])
def test_backend_rejects_an_allowed_ip_that_is_not_an_address_or_network(ip: str) -> None:
    with pytest.raises(ValidationError, match="allowed_ips"):
        ManifestBackend(name="ledger", ports=[8087], allowed_ips=[ip])


@pytest.mark.parametrize(
    "fields",
    [{"ports": []}, {"ports": [0]}, {"ports": [70000]}, {"name": "-bad"}, {"compose_file": "docker-compose.yaml"}],
)
def test_backend_rejects_invalid_fields(fields: dict) -> None:
    with pytest.raises(ValidationError):
        ManifestBackend.model_validate({"name": "ledger", "ports": [8087], **fields})


def test_manifest_bodies_accept_both_backend_forms() -> None:
    structured = {"name": "ledger", "ports": [8087], "host": "ledger.ns.svc"}

    init = ManifestInit(name="m1", agent="a", backends=["db:5432", structured])
    update = ManifestUpdate(backends=["db:5432", structured])

    assert init.backends == ["db:5432", ManifestBackend.model_validate(structured)]
    assert update.backends == init.backends
