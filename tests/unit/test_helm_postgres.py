# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Secret lookup branches that offline CI values renders cannot reach."""

import base64
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"


@pytest.mark.parametrize("password", ["nemo", "previously-generated-password", "", None])
def test_upgrade_reuses_lookup_password_or_rejects_missing_data(tmp_path: Path, password: str | None) -> None:
    """Stub only lookup so the real template must preserve or reject stored credentials."""
    if shutil.which("helm") is None:
        pytest.skip("helm is required")
    templates = tmp_path / "templates"
    templates.mkdir()
    (tmp_path / "Chart.yaml").write_text("apiVersion: v2\nname: postgres-test\nversion: 0.0.0\n")
    values = {
        "postgresql": {"enabled": True, "auth": {"password": None, "existingSecret": ""}},
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
    result = subprocess.run(
        ["helm", "template", "postgres-test", str(tmp_path), "--is-upgrade"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if password:
        assert result.returncode == 0, result.stderr
        assert yaml.safe_load(result.stdout)["stringData"]["password"] == password
    else:
        assert result.returncode != 0
        assert "existing PostgreSQL Secret must contain data.password" in result.stderr
