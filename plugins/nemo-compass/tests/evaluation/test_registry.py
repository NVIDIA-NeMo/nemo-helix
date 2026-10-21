# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from evaluation import cli, release
from evaluation.registry import load_registry


def test_loads_subject_with_shared_default(tmp_path: Path):
    toml = tmp_path / "t.toml"
    toml.write_text(
        'base_url = "https://shared"\n\n[nvq]\ntype = "intake"\nagent = "content-dedup"\nworkspace = "nvq"\n'
    )
    subjects = load_registry(toml)
    assert set(subjects) == {"nvq"}
    nvq = subjects["nvq"]
    assert nvq.type == "intake"
    assert nvq.config["agent"] == "content-dedup"
    assert nvq.config["base_url"] == "https://shared"  # shared default merged in


def test_per_subject_override_wins(tmp_path: Path):
    toml = tmp_path / "t.toml"
    toml.write_text('base_url = "https://shared"\n\n[local]\ntype = "intake"\nbase_url = "http://localhost:8000"\n')
    assert load_registry(toml)["local"].config["base_url"] == "http://localhost:8000"


def test_registry_contains_only_expected_analyzable_subjects() -> None:
    subjects = load_registry(cli.REGISTRY_PATH)

    assert set(subjects) == {
        "glamr",
        "kernel-factory",
        "nemo-oo-airline",
        "nvq",
        "tau2-airline",
        "tau2-retail",
        "tau2-telecom",
    }
    assert all(subject.type in ("benchmark", "intake") for subject in subjects.values())
    assert subjects["nvq"].config["agent"] == "content-dedup"


def test_glamr_stores_credential_environment_names_only() -> None:
    glamr = load_registry(cli.REGISTRY_PATH)["glamr"]

    assert glamr.config["auth_user_env"] == "GLAMR_INTAKE_USER"
    assert glamr.config["auth_password_env"] == "GLAMR_INTAKE_PASSWORD"
    assert "auth_user" not in glamr.config
    assert "auth_password" not in glamr.config


def test_nemo_oo_airline_is_an_intake_subject() -> None:
    subject = load_registry(cli.REGISTRY_PATH)["nemo-oo-airline"]

    assert subject.type == "intake"
    assert subject.config["agent"] == "nemo-oo-airline"


def test_tau2_telecom_uses_small_split() -> None:
    telecom = load_registry(cli.REGISTRY_PATH)["tau2-telecom"]

    assert telecom.type == "benchmark"
    assert telecom.config["domain"] == "telecom"
    assert telecom.config["task_split_name"] == "small"


def test_every_analyzable_subject_has_expected_state_pin() -> None:
    expected = {
        "glamr": "state-v8",
        "kernel-factory": "state-v11",
        "nemo-oo-airline": "state-v9",
        "nvq": "state-v7",
        "tau2-airline": "state-v6",
        "tau2-retail": "state-v10",
        "tau2-telecom": "state-v10",
    }

    subjects = load_registry(cli.REGISTRY_PATH)
    assert {name: release.pinned_ref(subjects[name]) for name in sorted(subjects)} == expected
