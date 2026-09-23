# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How `push` gets the registry credential, and hands it to crane and cosign.

It reads a Kubernetes Secret as its own ServiceAccount -- the route the signing key takes --
rather than having the Jobs launcher resolve a Secrets-service entry as the submitter, which would
make the operator's credential readable by everyone who can submit a build.

Neither local environment exercises the result: the minikube registry is anonymous, so a
credential that never reached crane looks exactly like one that did. An earlier version shipped in
that state, writing the credential under the host key ``""``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import stat
import tempfile
from pathlib import Path

import pytest
from kubernetes import client as k8s
from kubernetes.client.exceptions import ApiException
from nemo_builder_plugin.run.push import (
    DOCKERCONFIGJSON_KEY,
    CredentialError,
    _materialize_credential,
    _read_credential,
)

REGISTRY = "reg.example.com:5000"
NAMESPACE = "nhx-builds"
SECRET = "registry-push-credential"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the temp directory and `DOCKER_CONFIG` from leaking out of each test."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)


class FakeSecrets:
    """Stands in for `CoreV1Api`, and records what was read."""

    def __init__(self, data: dict[str, str] | None = None, *, error: ApiException | None = None) -> None:
        self.data = data
        self.error = error
        self.reads: list[tuple[str, str]] = []

    def read_namespaced_secret(self, name: str, namespace: str) -> k8s.V1Secret:
        self.reads.append((namespace, name))
        if self.error is not None:
            raise self.error
        return k8s.V1Secret(data=self.data)


def _encoded(document: object) -> dict[str, str]:
    """A Secret's `data`, as `kubectl create secret docker-registry` writes it."""
    return {DOCKERCONFIGJSON_KEY: base64.b64encode(json.dumps(document).encode()).decode()}


def _written() -> dict:
    return json.loads((Path(os.environ["DOCKER_CONFIG"]) / "config.json").read_text())


class TestReadingTheSecret:
    def test_it_reads_the_named_secret_in_the_build_namespace(self) -> None:
        docker_config = {"auths": {REGISTRY: {"auth": "eHg6eXk="}}}
        api = FakeSecrets(_encoded(docker_config))
        assert json.loads(_read_credential(api, namespace=NAMESPACE, name=SECRET)) == docker_config
        assert api.reads == [(NAMESPACE, SECRET)]

    def test_a_secret_it_may_not_read_is_refused(self) -> None:
        """A 403 means the RBAC grant is missing or names another Secret. Never push anonymously."""
        api = FakeSecrets(error=ApiException(status=403, reason="Forbidden"))
        with pytest.raises(CredentialError, match="403"):
            _read_credential(api, namespace=NAMESPACE, name=SECRET)

    def test_a_secret_without_a_docker_config_is_refused(self) -> None:
        api = FakeSecrets({"password": base64.b64encode(b"s3cret").decode()})
        with pytest.raises(CredentialError, match=DOCKERCONFIGJSON_KEY):
            _read_credential(api, namespace=NAMESPACE, name=SECRET)


class TestMaterializing:
    def test_it_is_written_as_is(self) -> None:
        docker_config = {"auths": {REGISTRY: {"auth": "eHg6eXk="}}}
        _materialize_credential(json.dumps(docker_config), registry=REGISTRY)
        assert _written() == docker_config

    def test_the_file_is_private_to_this_step(self) -> None:
        directory = _materialize_credential(json.dumps({"auths": {REGISTRY: {}}}), registry=REGISTRY)
        assert directory == os.environ["DOCKER_CONFIG"]
        assert stat.S_IMODE((Path(directory) / "config.json").stat().st_mode) == 0o600

    @pytest.mark.parametrize("value", ["robot:s3cret", "[]", '{"credsStore": "gcloud"}'])
    def test_anything_but_a_docker_config_with_auths_is_refused(self, value: str) -> None:
        with pytest.raises(CredentialError):
            _materialize_credential(value, registry=REGISTRY)
        assert "DOCKER_CONFIG" not in os.environ

    def test_no_entry_for_the_registry_is_said_out_loud(self, caplog: pytest.LogCaptureFixture) -> None:
        """Legitimate on an anonymous registry, so not refused -- but crane will push anonymously,
        and the log has to say so, or a credential that never reached crane stays invisible."""
        with caplog.at_level(logging.WARNING):
            _materialize_credential(json.dumps({"auths": {"elsewhere.example.com": {}}}), registry=REGISTRY)
        assert f"no entry for {REGISTRY}" in caplog.text

    def test_a_scheme_on_the_host_still_matches(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _materialize_credential(json.dumps({"auths": {f"https://{REGISTRY}/": {}}}), registry=REGISTRY)
        assert "no entry" not in caplog.text
