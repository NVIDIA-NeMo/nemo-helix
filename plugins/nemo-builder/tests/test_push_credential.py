# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How `push` hands the registry credential to crane and cosign.

Neither local environment exercised this: the minikube registry is anonymous, so a credential that
never reached crane looked exactly like one that did. The bare `user:password` form was in that
state -- written under the host key ``""``, which matches no registry.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import tempfile
from pathlib import Path

import pytest
from nemo_builder_plugin.run.push import CREDENTIAL_ENVVAR, CredentialError, _materialize_credential

REGISTRY = "reg.example.com:5000"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the temp directory and `DOCKER_CONFIG` from leaking out of each test."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)
    monkeypatch.delenv(CREDENTIAL_ENVVAR, raising=False)


def _written() -> dict:
    return json.loads((Path(os.environ["DOCKER_CONFIG"]) / "config.json").read_text())


class TestBareCredential:
    def test_it_is_bound_to_the_deployment_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The regression: this used to produce `{"auths": {"": ...}}`, and crane pushed anonymously."""
        monkeypatch.setenv(CREDENTIAL_ENVVAR, "robot:s3cret")
        _materialize_credential(REGISTRY)
        assert _written() == {"auths": {REGISTRY: {"auth": base64.b64encode(b"robot:s3cret").decode()}}}

    def test_a_password_may_contain_a_colon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(CREDENTIAL_ENVVAR, "robot:pa:ss")
        _materialize_credential(REGISTRY)
        assert base64.b64decode(_written()["auths"][REGISTRY]["auth"]) == b"robot:pa:ss"

    def test_with_no_registry_to_bind_to_it_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Refused rather than guessed: the only other hosts on hand are ones a caller chose."""
        monkeypatch.setenv(CREDENTIAL_ENVVAR, "robot:s3cret")
        with pytest.raises(CredentialError, match="default_registry"):
            _materialize_credential(None)
        assert "DOCKER_CONFIG" not in os.environ

    def test_a_value_that_is_neither_form_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(CREDENTIAL_ENVVAR, "just-a-token")
        with pytest.raises(CredentialError, match="neither"):
            _materialize_credential(REGISTRY)


class TestDockerConfig:
    def test_it_is_used_as_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """It already names its hosts, so the deployment registry does not rebind it."""
        docker_config = {"auths": {"elsewhere.example.com": {"auth": "eHg6eXk="}}}
        monkeypatch.setenv(CREDENTIAL_ENVVAR, json.dumps(docker_config))
        _materialize_credential(REGISTRY)
        assert _written() == docker_config


class TestWhereItLands:
    def test_the_file_is_private_to_this_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(CREDENTIAL_ENVVAR, "robot:s3cret")
        directory = _materialize_credential(REGISTRY)
        assert directory is not None and directory == os.environ["DOCKER_CONFIG"]
        assert stat.S_IMODE((Path(directory) / "config.json").stat().st_mode) == 0o600

    def test_no_credential_means_no_docker_config(self) -> None:
        assert _materialize_credential(REGISTRY) is None
        assert "DOCKER_CONFIG" not in os.environ
