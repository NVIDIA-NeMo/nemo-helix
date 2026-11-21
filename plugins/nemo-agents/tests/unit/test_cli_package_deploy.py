# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo agents package-deploy`` and its preflight helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nemo_agents_plugin import package_deploy
from nemo_agents_plugin.cli import AgentsCLI
from nemo_agents_plugin.package_deploy import (
    PreflightError,
    TargetConnection,
    TargetInfo,
    connect_to_target,
    registry_has_credentials,
    registry_host,
    resolve_contract_version,
    resolve_platform,
    saved_context_for,
    validate_registry,
)
from nemo_helix_plugin.client.client import NemoClient
from typer.testing import CliRunner

TARGET = "https://nemo.example.com"
REGISTRY = "nvcr.io/my-org/my-team"
LOCAL_TAG = "triage-abc123:1.0"

_CONFIG = """
current_context: local
clusters:
  - name: local
    base_url: http://localhost:8080
  - name: hosted
    base_url: https://nemo.example.com/
users:
  - name: local
    type: no-auth
  - name: hosted
    type: no-auth
contexts:
  - name: local
    cluster: local
    user: local
  - name: hosted
    cluster: hosted
    user: hosted
"""


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(_CONFIG)
    monkeypatch.setenv("NHX_CONFIG_FILE", str(path))
    return path


@pytest.fixture
def agent_config(tmp_path: Path) -> Path:
    path = tmp_path / "agent" / "agent.yaml"
    path.parent.mkdir()
    path.write_text("config_format: nemo-agents-spec-v1\nname: triage\n")
    return path


class TestRegistry:
    def test_scheme_is_rejected_with_the_bare_form(self) -> None:
        with pytest.raises(PreflightError, match=f"Use: --registry {REGISTRY}"):
            validate_registry(f"https://{REGISTRY}")

    def test_trailing_slash_is_dropped(self) -> None:
        assert validate_registry(f"{REGISTRY}/") == REGISTRY

    @pytest.mark.parametrize(
        ("registry", "host"),
        [
            (REGISTRY, "nvcr.io"),
            ("nvcr.io", "nvcr.io"),
            ("localhost:5000/agents", "localhost:5000"),
            ("myuser", "index.docker.io"),
            ("myuser/agents", "index.docker.io"),
        ],
    )
    def test_registry_host(self, registry: str, host: str) -> None:
        assert registry_host(registry) == host

    @pytest.mark.parametrize(
        ("docker_config", "expected"),
        [
            ({"auths": {"nvcr.io": {}}}, True),
            ({"auths": {"https://nvcr.io": {"auth": "x"}}}, True),
            ({"auths": {"ghcr.io": {}}}, False),
            ({}, False),
            ({"credHelpers": {"nvcr.io": "gcloud"}}, True),
        ],
    )
    def test_credentials_from_docker_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, docker_config: dict[str, Any], expected: bool
    ) -> None:
        (tmp_path / "config.json").write_text(json.dumps(docker_config))
        monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
        assert registry_has_credentials(REGISTRY) is expected

    @pytest.mark.parametrize(
        ("listed", "expected"), [({"https://nvcr.io": "$oauthtoken"}, True), ({}, False), (None, None)]
    )
    def test_credentials_from_credential_store(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        listed: dict[str, str] | None,
        expected: bool | None,
    ) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"credsStore": "desktop", "auths": {"nvcr.io": {}}}))
        monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
        hosts = None if listed is None else {package_deploy._normalize_auth_key(key) for key in listed}
        with patch.object(package_deploy, "_credential_store_hosts", return_value=hosts):
            assert registry_has_credentials(REGISTRY) is expected

    def test_local_registry_needs_no_credentials(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
        assert registry_has_credentials("localhost:5000/agents") is True


class TestResolveContractVersion:
    @pytest.mark.parametrize("server", ["0.10.0", "v0.10.0"])
    def test_defaults_to_the_targets_release(self, server: str) -> None:
        assert resolve_contract_version(server, None, target_url=TARGET) == "0.10.0"

    @pytest.mark.parametrize("server", ["dev", None, "0.11.0.dev0"])
    def test_unreleased_target_requires_a_version(self, server: str | None) -> None:
        with pytest.raises(PreflightError, match="Pass --nemo-version"):
            resolve_contract_version(server, None, target_url=TARGET)

    def test_unreleased_target_takes_the_requested_release(self) -> None:
        assert resolve_contract_version("dev", "0.10.0", target_url=TARGET) == "0.10.0"

    def test_mismatch_with_a_released_target_is_refused(self) -> None:
        with pytest.raises(PreflightError, match="does not match nemo-helix 0.10.0"):
            resolve_contract_version("0.10.0", "0.9.0", target_url=TARGET)

    def test_unpublished_requested_version_is_refused(self) -> None:
        with pytest.raises(PreflightError, match="--nemo-version 0.11.0.dev0"):
            resolve_contract_version("dev", "0.11.0.dev0", target_url=TARGET)


class TestResolvePlatform:
    def test_requested_overrides_the_target(self) -> None:
        assert resolve_platform("linux/arm64", "linux/amd64", target_url=TARGET) == "linux/amd64"

    def test_defaults_to_the_target(self) -> None:
        assert resolve_platform("linux/arm64", None, target_url=TARGET) == "linux/arm64"

    def test_required_when_the_target_reports_none(self) -> None:
        with pytest.raises(PreflightError, match="Pass --platform"):
            resolve_platform(None, None, target_url=TARGET)


class TestConnectToTarget:
    def test_saved_context_matches_the_url_ignoring_the_trailing_slash(self, config_file: Path) -> None:
        assert saved_context_for(TARGET) == "hosted"
        assert saved_context_for("https://other.example.com") is None

    def test_reuses_a_valid_saved_login_even_when_env_points_elsewhere(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NHX_BASE_URL", "http://localhost:8080")
        before = config_file.read_bytes()
        with (
            patch.object(package_deploy, "discover_nhx_config", return_value=MagicMock(auth_enabled=True)),
            patch.object(package_deploy, "ensure_valid_token", return_value=True),
            patch.object(package_deploy, "authenticate_with_oidc") as login,
        ):
            connection = connect_to_target(TARGET, "default", http_client=MagicMock())

        login.assert_not_called()
        assert connection.description == "saved context 'hosted'"
        assert connection.client.base_url == TARGET
        assert config_file.read_bytes() == before

    def test_logs_in_for_this_run_only_without_writing_config(self, config_file: Path) -> None:
        before = config_file.read_bytes()
        oidc = MagicMock(auth_enabled=True)
        login_result = MagicMock(access_token="token", refresh_token="refresh", expires_at=None, token_broker_url=None)
        settings = MagicMock(
            token_endpoint="https://idp/token",
            client_id="cli",
            refresh_scope=None,
            bearer_token_source="access_token",
            refresh_kind="provider",
        )
        with (
            patch.object(package_deploy, "discover_nhx_config", return_value=oidc),
            patch.object(package_deploy, "ensure_valid_token", return_value=False),
            patch.object(package_deploy, "authenticate_with_oidc", return_value=login_result) as login,
            patch.object(package_deploy, "resolve_refresh_settings", return_value=settings),
        ):
            connection = connect_to_target(TARGET, "default", http_client=MagicMock(), no_browser=True)

        login.assert_called_once_with(oidc_config=oidc, no_browser=True)
        assert connection.client.base_url == TARGET
        assert "not saved" in connection.description
        assert config_file.read_bytes() == before

    def test_auth_disabled_target_needs_no_login(self, config_file: Path) -> None:
        with (
            patch.object(package_deploy, "discover_nhx_config", return_value=MagicMock(auth_enabled=False)),
            patch.object(package_deploy, "authenticate_with_oidc") as login,
        ):
            connection = connect_to_target(TARGET, "default", http_client=MagicMock())

        login.assert_not_called()
        assert connection.client.base_url == TARGET


Handler = Callable[[httpx.Request], httpx.Response]


def _target_handler(requests: list[httpx.Request], *, agent_exists: bool = True) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if request.method == "GET" and path.endswith("/agents/triage-agent"):
            if not agent_exists:
                return httpx.Response(404, json={"detail": "Agent 'triage-agent' not found"})
            return httpx.Response(200, json={"name": "triage-agent", "workspace": "default"})
        if request.method == "POST" and path.endswith("/deployments"):
            return httpx.Response(201, json={"name": "triage-agent-d1", "status": "pending"})
        if request.method == "GET" and path.endswith("/deployments/triage-agent-d1"):
            return httpx.Response(200, json={"name": "triage-agent-d1", "status": "running"})
        return httpx.Response(500, json={"detail": f"unexpected {request.method} {path}"})

    return handler


class _Harness:
    def __init__(self, handler: Handler, target_info: TargetInfo, has_credentials: bool | None) -> None:
        client = NemoClient(
            base_url=TARGET,
            workspace="default",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url=TARGET),
        )
        self.connect = patch.object(
            package_deploy, "connect_to_target", return_value=TargetConnection(client, "saved context 'hosted'")
        )
        self.info = patch.object(package_deploy, "fetch_cluster_info", return_value=target_info)
        self.credentials = patch.object(package_deploy, "registry_has_credentials", return_value=has_credentials)
        self.build = patch("nemo_agents_plugin.container.builder.build_fabric_agent_image", return_value=LOCAL_TAG)
        self.push = patch(
            "nemo_agents_plugin.container.publisher.docker_push",
            side_effect=lambda **kwargs: kwargs["push_tag"],
        )


@pytest.fixture
def run_package_deploy(config_file: Path, agent_config: Path) -> Iterator[Callable[..., Any]]:
    def run(
        *extra: str,
        handler: Handler,
        target_info: TargetInfo = TargetInfo("0.10.0", "linux/amd64"),
        has_credentials: bool | None = True,
        registry: str = REGISTRY,
        config: Path = agent_config,
    ) -> tuple[Any, MagicMock, MagicMock, MagicMock]:
        harness = _Harness(handler, target_info, has_credentials)
        with (
            harness.connect as connect,
            harness.info,
            harness.credentials,
            harness.build as build,
            harness.push as push,
            patch("nemo_agents_plugin.cli.sleep"),
        ):
            result = CliRunner().invoke(
                AgentsCLI().get_cli(),
                [
                    "package-deploy",
                    "--agent",
                    "triage-agent",
                    "--agent-config",
                    str(config),
                    "--target-url",
                    f"{TARGET}/",
                    "--workspace",
                    "default",
                    "--registry",
                    registry,
                    *extra,
                ],
            )
        return result, connect, build, push

    yield run


class TestPackageDeployCommand:
    def test_happy_path_builds_pushes_and_deploys_to_the_target_only(
        self, run_package_deploy: Callable[..., Any], config_file: Path
    ) -> None:
        before = config_file.read_bytes()
        requests: list[httpx.Request] = []

        result, connect, build, push = run_package_deploy(handler=_target_handler(requests))

        assert result.exit_code == 0, result.output
        assert connect.call_args.args[0] == TARGET
        build_kwargs = build.call_args.kwargs
        assert build_kwargs["platforms"] == ["linux/amd64"]
        assert build_kwargs["contract_version"] == "0.10.0"
        push_tag = push.call_args.kwargs["push_tag"]
        assert push_tag.startswith(f"{REGISTRY}/triage-abc123:1.0-")
        deploy_body = json.loads(next(r for r in requests if r.method == "POST").read())
        assert deploy_body == {"agent": "triage-agent", "deployment_mode": "k8s", "image": push_tag}
        assert {request.url.host for request in requests} == {"nemo.example.com"}
        assert config_file.read_bytes() == before

    def test_explicit_platform_and_version_override_an_unreleased_target(
        self, run_package_deploy: Callable[..., Any]
    ) -> None:
        result, _, build, _ = run_package_deploy(
            "--platform",
            "linux/arm64",
            "--nemo-version",
            "0.10.0",
            handler=_target_handler([]),
            target_info=TargetInfo("dev", None),
        )

        assert result.exit_code == 0, result.output
        assert build.call_args.kwargs["platforms"] == ["linux/arm64"]
        assert build.call_args.kwargs["contract_version"] == "0.10.0"

    def test_unique_tag_per_build(self, run_package_deploy: Callable[..., Any]) -> None:
        with patch("nemo_agents_plugin.cli.datetime") as clock:
            clock.now.return_value.strftime.side_effect = ["20261120120000", "20261120120501"]
            _, _, _, first = run_package_deploy(handler=_target_handler([]))
            first_tag = first.call_args.kwargs["push_tag"]
            _, _, _, second = run_package_deploy(handler=_target_handler([]))

        assert first_tag != second.call_args.kwargs["push_tag"]

    def test_scheme_in_registry_fails_before_anything_else(self, run_package_deploy: Callable[..., Any]) -> None:
        result, connect, build, _ = run_package_deploy(handler=_target_handler([]), registry=f"https://{REGISTRY}")

        assert result.exit_code == 2
        assert f"--registry {REGISTRY}" in result.output
        connect.assert_not_called()
        build.assert_not_called()

    def test_missing_registry_credentials_fail_with_the_login_hint(
        self, run_package_deploy: Callable[..., Any]
    ) -> None:
        result, connect, build, _ = run_package_deploy(handler=_target_handler([]), has_credentials=False)

        assert result.exit_code == 1
        assert "docker login nvcr.io -u '$oauthtoken'" in result.output
        connect.assert_not_called()
        build.assert_not_called()

    def test_unknown_agent_fails_before_building(self, run_package_deploy: Callable[..., Any]) -> None:
        result, _, build, _ = run_package_deploy(handler=_target_handler([], agent_exists=False))

        assert result.exit_code == 1
        assert "Agent 'triage-agent' not found in workspace 'default' on https://nemo.example.com" in result.output
        build.assert_not_called()

    def test_unreleased_target_without_a_version_fails_before_building(
        self, run_package_deploy: Callable[..., Any]
    ) -> None:
        result, _, build, _ = run_package_deploy(
            handler=_target_handler([]), target_info=TargetInfo("dev", "linux/amd64")
        )

        assert result.exit_code == 1
        assert "Pass --nemo-version" in result.output
        build.assert_not_called()

    def test_target_without_architecture_needs_platform(self, run_package_deploy: Callable[..., Any]) -> None:
        result, _, build, _ = run_package_deploy(handler=_target_handler([]), target_info=TargetInfo("0.10.0", None))

        assert result.exit_code == 1
        assert "Pass --platform" in result.output
        build.assert_not_called()

    def test_nat_config_is_refused(self, run_package_deploy: Callable[..., Any], tmp_path: Path) -> None:
        nat = tmp_path / "workflow.yaml"
        nat.write_text("config_format: nat-workflow-v1\nworkflow:\n  _type: react_agent\n")

        result, connect, build, _ = run_package_deploy(handler=_target_handler([]), config=nat)

        assert result.exit_code == 1
        assert "Fabric agents only" in result.output
        connect.assert_not_called()
        build.assert_not_called()
