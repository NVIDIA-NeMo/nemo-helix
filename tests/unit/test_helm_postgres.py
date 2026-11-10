# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"


def _render(*args: str, chart: Path = HELM_DIR, success: bool = True) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required")
    # Avoid the unrelated encryption-key lookup guard during offline upgrades.
    shared_args = ["--set", "envFromSecret=test-api-env"] if chart == HELM_DIR else []
    result = subprocess.run(
        ["helm", "template", "postgres-test", str(chart), *shared_args, *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if not success:
        assert result.returncode != 0
        assert "PostgreSQL Secret" in result.stderr
        return []
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _postgres_secret(docs: list[dict]) -> dict:
    return next(doc for doc in docs if doc["kind"] == "Secret" and doc["metadata"]["name"].endswith("-postgres"))


def test_fresh_install_password_is_random() -> None:
    first = _postgres_secret(_render())["stringData"]["password"]
    second = _postgres_secret(_render())["stringData"]["password"]
    assert len(first) == len(second) == 32
    assert first.isalnum() and second.isalnum()
    assert first != second


def test_explicit_password_is_preserved_on_upgrade() -> None:
    docs = _render("--is-upgrade", "--set-string", "postgresql.auth.password=explicit-password")
    assert _postgres_secret(docs)["stringData"]["password"] == "explicit-password"


def test_missing_upgrade_secret_fails_instead_of_regenerating_password() -> None:
    _render("--is-upgrade", success=False)


@pytest.mark.parametrize("password", ["nemo", "previously-generated-password", "", None])
def test_upgrade_reuses_lookup_password_or_rejects_missing_data(tmp_path: Path, password: str | None) -> None:
    # Offline Helm cannot query Secrets. Stub only the lookup expression in an
    # isolated chart to exercise the production template's upgrade branch.
    templates = tmp_path / "templates"
    templates.mkdir()
    (tmp_path / "Chart.yaml").write_text("apiVersion: v2\nname: postgres-test\nversion: 0.0.0\n")
    values = {
        "postgresql": {"enabled": True, "auth": {"password": "", "existingSecret": ""}},
        "lookupResult": {
            "data": {"password": base64.b64encode(password.encode()).decode()} if password is not None else None
        },
    }
    (tmp_path / "values.yaml").write_text(yaml.safe_dump(values))
    source = (HELM_DIR / "templates/postgres/postgres-secret.yaml").read_text()
    lookup = 'lookup "v1" "Secret" .Release.Namespace $secretName'
    assert source.count(lookup) == 1
    (templates / "secret.yaml").write_text(source.replace(lookup, ".Values.lookupResult"))
    (templates / "_helpers.tpl").write_text(
        '{{- define "nemo-common.postgresql.fullname" -}}test-postgres{{- end -}}\n'
        '{{- define "nemo-helix.labels" -}}app: test{{- end -}}\n'
    )
    docs = _render("--is-upgrade", chart=tmp_path, success=bool(password))
    if password:
        assert _postgres_secret(docs)["stringData"]["password"] == password


def test_existing_secret_is_used_by_database_and_clients() -> None:
    docs = _render(
        "--is-upgrade",
        "--set",
        "postgresql.auth.existingSecret=custom-db",
        "--set",
        "postgresql.auth.existingSecretPasswordKey=db-password",
    )
    assert not any(doc["kind"] == "Secret" and doc["metadata"]["name"].endswith("-postgres") for doc in docs)
    references = []
    for doc in docs:
        if doc["kind"] not in {"Deployment", "StatefulSet"}:
            continue
        for container in doc["spec"]["template"]["spec"]["containers"]:
            for env in container.get("env", []):
                ref = env.get("valueFrom", {}).get("secretKeyRef", {})
                if ref.get("name") == "custom-db":
                    references.append(ref)
    assert references == [{"name": "custom-db", "key": "db-password"}] * 3


@pytest.mark.parametrize(
    "settings",
    [(), ("--set", "networkPolicies.postgresql.enabled=false"), ("--set", "postgresql.enabled=false")],
)
def test_postgres_policy_is_conditional(settings: tuple[str, ...]) -> None:
    args = () if not settings else ("--set", "networkPolicies.enabled=true")
    docs = _render(*args, *settings)
    assert not any(
        doc["kind"] == "NetworkPolicy" and doc["metadata"]["name"].endswith("-postgres-ingress") for doc in docs
    )


def test_postgres_policy_allows_only_same_release_database_clients() -> None:
    docs = _render(
        "--set",
        "networkPolicies.enabled=true",
        "--set",
        "api.componentLabel=legacy-api",
        "--set",
        "core.controller.componentLabel=legacy-controller",
        "--set",
        "postgresql.service.port=15432",
    )
    policy = next(
        doc for doc in docs if doc["kind"] == "NetworkPolicy" and doc["metadata"]["name"].endswith("-postgres-ingress")
    )
    spec = policy["spec"]
    labels = {"app.kubernetes.io/name": "nemo-helix", "app.kubernetes.io/instance": "postgres-test"}
    assert spec["podSelector"] == {"matchLabels": {**labels, "app.kubernetes.io/component": "postgres"}}
    assert spec["policyTypes"] == ["Ingress"]
    assert spec["ingress"] == [
        {
            "from": [
                {"podSelector": {"matchLabels": {**labels, "app.kubernetes.io/component": component}}}
                for component in ("legacy-api", "legacy-controller")
            ],
            # Policies match the destination pod port, not the Service port.
            "ports": [{"protocol": "TCP", "port": 5432}],
        }
    ]


def test_postgres_policy_supports_extra_ingress_without_controller(tmp_path: Path) -> None:
    extra = {
        "from": [{"podSelector": {"matchLabels": {"app": "backup"}}}],
        "ports": [{"protocol": "TCP", "port": 5432}],
    }
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        yaml.safe_dump(
            {"networkPolicies": {"enabled": True, "postgresql": {"extraIngress": [extra]}}, "core": {"enabled": False}}
        )
    )
    docs = _render("-f", str(overrides))
    policy = next(
        doc for doc in docs if doc["kind"] == "NetworkPolicy" and doc["metadata"]["name"].endswith("-postgres-ingress")
    )
    assert len(policy["spec"]["ingress"][0]["from"]) == 1
    assert policy["spec"]["ingress"][1] == extra
