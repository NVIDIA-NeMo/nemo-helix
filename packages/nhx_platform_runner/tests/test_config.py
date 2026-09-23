# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
from nhx.platform_runner import registry
from nhx.platform_runner.config import (
    DEFAULT_PLATFORM_BIND_HOST,
    DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS,
    HelixAppConfig,
    ResolvedRunConfiguration,
    apply_run_environment,
    default_config_path,
    resolve_run_configuration,
)


@pytest.fixture(autouse=True)
def clear_registry_caches() -> None:
    registry.get_available_services.cache_clear()
    registry.get_available_controllers.cache_clear()


def _make_config(
    *,
    services: set[str] | None = None,
    controllers: set[str] | None = None,
    sidecars: set[str] | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    socket_path: str | None = None,
    config_path: str = "/nonexistent/nhx-test-config.yaml",
) -> ResolvedRunConfiguration:
    return ResolvedRunConfiguration(
        services=services if services is not None else {"auth", "entities"},
        controllers=controllers if controllers is not None else {"jobs"},
        sidecars=sidecars if sidecars is not None else set(),
        host=host,
        port=port,
        socket_path=socket_path,
        config_path=config_path,
    )


def resolve(**kwargs):
    return resolve_run_configuration(HelixAppConfig(**kwargs))


def test_default_config_path_points_to_bundled_local_config():
    path = default_config_path()

    assert path.endswith(("nhx/platform_runner/config/local.yaml", "nemo_helix/services/runner/config/local.yaml"))


def test_platform_app_config_keeps_sequence_fields_simple():
    config = HelixAppConfig(
        services=["models"],
        controllers=[],
        sidecars=["adapters"],
    )

    assert config.services == ["models"]
    assert config.controllers == []
    assert config.sidecars == ["adapters"]


def test_platform_app_config_derives_instance_paths_from_roots(tmp_path: Path):
    config = HelixAppConfig(scope="dev", state_root=tmp_path / "state", runtime_root=tmp_path / "run")

    assert config.state_dir() == tmp_path / "state" / "instances" / "dev"
    assert config.runtime_dir() == tmp_path / "run" / "dev"
    assert config.socket_file_path() == tmp_path / "run" / "dev" / "nemo-helix.sock"
    assert config.log_file_path() == tmp_path / "state" / "instances" / "dev" / "services.log"


def test_platform_app_config_runtime_dir_defaults_to_explicit_socket_parent(tmp_path: Path):
    config = HelixAppConfig(socket_path=tmp_path / "custom.sock")

    assert config.runtime_dir() == tmp_path
    assert config.socket_file_path() == tmp_path / "custom.sock"


def test_platform_app_config_uses_explicit_log_path(tmp_path: Path):
    config = HelixAppConfig(state_root=tmp_path / "state", log_path=tmp_path / "logs" / "nemo.log")

    assert config.log_file_path() == tmp_path / "logs" / "nemo.log"


def test_platform_app_config_defaults_keep_alive_timeout():
    config = HelixAppConfig()

    assert config.keep_alive_timeout_seconds == DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS


@pytest.mark.parametrize("timeout_seconds", [0, -1, 1.5, True])
def test_platform_app_config_rejects_invalid_keep_alive_timeout(timeout_seconds: Any):
    with pytest.raises(ValueError, match="keep_alive_timeout_seconds must be greater than 0"):
        HelixAppConfig(keep_alive_timeout_seconds=timeout_seconds)


def test_platform_app_config_rejects_relative_socket_path():
    with pytest.raises(ValueError, match="UDS socket path must be absolute"):
        HelixAppConfig(socket_path="relative/path")


def test_platform_app_config_rejects_relative_state_root():
    with pytest.raises(ValueError, match="state root must be absolute"):
        HelixAppConfig(state_root="relative/path")


def test_platform_app_config_rejects_relative_runtime_root():
    with pytest.raises(ValueError, match="runtime root must be absolute"):
        HelixAppConfig(runtime_root="relative/path")


def test_platform_app_config_rejects_relative_log_path():
    with pytest.raises(ValueError, match="log path must be absolute"):
        HelixAppConfig(log_path="relative/path")


def test_resolve_run_configuration_accepts_platform_app_config():
    resolved = resolve_run_configuration(
        HelixAppConfig(
            services=["auth"],
            controllers=[],
            host="127.0.0.1",
            port=9090,
        )
    )

    assert resolved.services == {"auth"}
    assert resolved.controllers == set()
    assert resolved.host == "127.0.0.1"
    assert resolved.port == 9090
    assert resolved.keep_alive_timeout_seconds == DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS


def test_resolve_run_configuration_preserves_keep_alive_timeout():
    resolved = resolve(keep_alive_timeout_seconds=12)

    assert resolved.keep_alive_timeout_seconds == 12


def test_no_arguments_defaults_to_all_services_and_default_controllers():
    resolved = resolve()

    assert resolved.host == DEFAULT_PLATFORM_BIND_HOST
    assert resolved.services.issuperset(
        {
            "auth",
            "entities",
            "files",
            "inference-gateway",
            "intake",
            "jobs",
            "models",
            "secrets",
            "safe-synthesizer",
        }
    )
    # Default controllers include all available controllers (core + any installed plugins).
    assert resolved.controllers.issuperset({"jobs", "models", "entities"})


def test_service_group_all_does_not_start_controllers():
    """Helm API pods use --service-group=all with no controllers; do not auto-start them."""
    resolved = resolve(service_group="all")

    assert "entities" in resolved.services
    assert resolved.controllers == set()


def test_service_group_core_resolves_core_services_only():
    resolved = resolve(service_group="core")

    assert resolved.services == {
        "auth",
        "entities",
        "files",
        "inference-gateway",
        "jobs",
        "models",
        "secrets",
    }
    assert resolved.controllers == set()


def test_controller_group_all_resolves_default_controllers():
    resolved = resolve(controller_group="all")

    assert resolved.services == set()
    assert resolved.controllers.issuperset({"jobs", "models", "entities"})


def test_services_and_service_group_are_mutually_exclusive():
    with pytest.raises(ValueError, match="--services cannot be combined with --service-group"):
        resolve(services=["auth"], service_group="all")


def test_controllers_and_controller_group_are_mutually_exclusive():
    with pytest.raises(ValueError, match="--controllers cannot be combined with --controller-group"):
        resolve(controllers=["jobs"], controller_group="all")


def test_invalid_service_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown services: nope"):
        resolve(services=["nope"])


def test_extra_services_are_available_for_resolution():
    with pytest.raises(ValueError, match="Unknown services: custom-service"):
        resolve(services=["custom-service"])


def test_resolve_rejects_relative_socket_path():
    with pytest.raises(ValueError, match="UDS socket path must be absolute"):
        resolve(socket_path="relative.sock")


def test_resolve_preserves_absolute_socket_path():
    resolved = resolve(socket_path="/tmp/nemo-helix.sock")

    assert resolved.socket_path == "/tmp/nemo-helix.sock"


# ---------------------------------------------------------------------------
# Topology regression tests for apply_run_environment
#
# The revert of PR #15 was caused by apply_run_environment unconditionally
# setting NHX_BASE_URL to localhost, which broke k8s controllers that need
# NHX_BASE_URL to point to the API service pod (set by Helm).
#
# These tests inject a plain dict instead of touching os.environ, so they
# cannot leak state to other tests in the suite.
# ---------------------------------------------------------------------------


class TestApplyRunEnvStandalone:
    """Standalone mode: env vars are NOT pre-set. apply_run_environment should populate them."""

    def test_sets_base_url_when_not_present(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080), env=env)
        assert env["NHX_BASE_URL"] == "http://127.0.0.1:8080"

    def test_config_file_gateway_base_url_seeds_base_url(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
platform:
  base_url: "https://nemo-gateway:8080"
""",
            encoding="utf-8",
        )
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080, config_path=str(config_path)), env=env)
        assert env["NHX_BASE_URL"] == "https://nemo-gateway:8080"
        assert env["NHX_SERVICE_HOST"] == "127.0.0.1"

    def test_sets_uds_base_url_when_socket_path_is_present(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(socket_path="/tmp/nemo-helix.sock"), env=env)
        assert env["NHX_BASE_URL"] == "unix:///tmp/nemo-helix.sock"

    def test_sets_embedded_pdp_base_url_from_base_url(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=9090), env=env)
        assert env["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "http://127.0.0.1:9090"

    def test_embedded_pdp_base_url_uses_resolved_base_url_when_auth_config_is_static(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
platform:
  base_url: "https://nemo-gateway:8080"
auth:
  policy_decision_point_base_url: "http://127.0.0.1:8080"
""",
            encoding="utf-8",
        )
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=59007, config_path=str(config_path)), env=env)
        assert env["NHX_BASE_URL"] == "https://nemo-gateway:59007"
        assert env["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "https://nemo-gateway:59007"

    def test_sets_service_host_when_not_present(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080), env=env)
        assert env["NHX_SERVICE_HOST"] == "127.0.0.1"

    def test_sets_service_port_when_not_present(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=9090), env=env)
        assert env["NHX_SERVICE_PORT"] == "9090"

    def test_normalizes_ipv4_wildcard_to_loopback(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080), env=env)
        assert env["NHX_SERVICE_HOST"] == "127.0.0.1"

    def test_normalizes_ipv6_wildcard_to_loopback(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="::", port=8080), env=env)
        assert env["NHX_SERVICE_HOST"] == "::1"

    def test_ipv6_loopback_brackets_in_url(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="::", port=8080), env=env)
        assert env["NHX_BASE_URL"] == "http://[::1]:8080"

    def test_sets_services_and_controllers(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(services={"auth", "files"}, controllers={"jobs"}), env=env)
        assert env["NHX_SERVICES"] == "auth,files"
        assert env["NHX_CONTROLLERS"] == "jobs"

    def test_clears_empty_sidecars(self):
        env: dict[str, str] = {"NHX_SIDECARS": "old-value"}
        apply_run_environment(_make_config(sidecars=set()), env=env)
        assert "NHX_SIDECARS" not in env

    def test_quickstart_platform_config_backfills_dev_secret_key_creation(self, tmp_path: Path):
        config_path = tmp_path / "platform-config.yaml"
        config_path.write_text("secrets: {}\n", encoding="utf-8")

        env: dict[str, str] = {}
        apply_run_environment(_make_config(config_path=str(config_path)), env=env)

        assert env["NHX_SECRETS_ALLOW_KEY_CREATION"] == "1"
        assert env["NHX_SECRETS_LOCAL_KEY_CREATION_PATH"] == "/data/nhx-encryption-key.txt"

    def test_quickstart_platform_config_preserves_explicit_secret_key_creation_false(self, tmp_path: Path):
        config_path = tmp_path / "platform-config.yaml"
        config_path.write_text("secrets:\n  allow_key_creation: false\n", encoding="utf-8")

        env: dict[str, str] = {}
        apply_run_environment(_make_config(config_path=str(config_path)), env=env)

        assert "NHX_SECRETS_ALLOW_KEY_CREATION" not in env
        assert "NHX_SECRETS_LOCAL_KEY_CREATION_PATH" not in env

    def test_quickstart_platform_config_preserves_existing_secret_env(self, tmp_path: Path):
        config_path = tmp_path / "platform-config.yaml"
        config_path.write_text("secrets: {}\n", encoding="utf-8")

        env: dict[str, str] = {"NHX_SECRETS_ALLOW_KEY_CREATION": "0"}
        apply_run_environment(_make_config(config_path=str(config_path)), env=env)

        assert env["NHX_SECRETS_ALLOW_KEY_CREATION"] == "0"
        assert "NHX_SECRETS_LOCAL_KEY_CREATION_PATH" not in env

    def test_quickstart_platform_config_preserves_named_secret_provider(self, tmp_path: Path):
        config_path = tmp_path / "platform-config.yaml"
        config_path.write_text(
            """
secrets:
  encryption:
    current_provider: local_v1
    providers:
      secret_key:
        local_v1:
          from_env: NHX_SECRETS_DEFAULT_ENCRYPTION_KEY
""".lstrip(),
            encoding="utf-8",
        )

        env: dict[str, str] = {}
        apply_run_environment(_make_config(config_path=str(config_path)), env=env)

        assert "NHX_SECRETS_ALLOW_KEY_CREATION" not in env
        assert "NHX_SECRETS_LOCAL_KEY_CREATION_PATH" not in env

    def test_non_quickstart_config_does_not_backfill_dev_secret_key_creation(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("secrets: {}\n", encoding="utf-8")

        env: dict[str, str] = {}
        apply_run_environment(_make_config(config_path=str(config_path)), env=env)

        assert "NHX_SECRETS_ALLOW_KEY_CREATION" not in env
        assert "NHX_SECRETS_LOCAL_KEY_CREATION_PATH" not in env


class TestApplyRunEnvDeployed:
    """Deployed mode: env vars are pre-set by Helm/k8s. apply_run_environment must NOT overwrite them.

    This is the regression that caused the revert of PR #15. Controllers in k8s
    run in separate pods where NHX_BASE_URL must point to the API service, not localhost.
    """

    def test_preserves_existing_base_url(self):
        env: dict[str, str] = {"NHX_BASE_URL": "http://nemo-helix-api:8080"}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080), env=env)
        assert env["NHX_BASE_URL"] == "http://nemo-helix-api:8080"

    def test_preserves_existing_embedded_pdp_base_url(self):
        env: dict[str, str] = {
            "NHX_BASE_URL": "http://nemo-helix-api:8080",
            "NHX_AUTH_POLICY_DECISION_POINT_BASE_URL": "http://nemo-auth:8080",
        }
        apply_run_environment(_make_config(host="0.0.0.0", port=8080), env=env)
        assert env["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "http://nemo-auth:8080"

    def test_preserves_existing_service_host(self):
        env: dict[str, str] = {"NHX_SERVICE_HOST": "nemo-helix-api"}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080), env=env)
        assert env["NHX_SERVICE_HOST"] == "nemo-helix-api"

    def test_preserves_existing_service_port(self):
        env: dict[str, str] = {"NHX_SERVICE_PORT": "443"}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080), env=env)
        assert env["NHX_SERVICE_PORT"] == "443"

    def test_derives_base_url_from_effective_host_port(self):
        """If Helm sets host/port but not NHX_BASE_URL, derive it from the effective values."""
        env: dict[str, str] = {
            "NHX_SERVICE_HOST": "nemo-helix-api",
            "NHX_SERVICE_PORT": "9090",
        }
        apply_run_environment(_make_config(host="0.0.0.0", port=8080), env=env)
        assert env["NHX_BASE_URL"] == "http://nemo-helix-api:9090"


class TestApplyRunEnvConfigBaseUrl:
    """Standalone mode with an explicit platform.base_url in the config file.

    The config value's *host* must seed NHX_BASE_URL (instead of the
    bind-derived loopback default) so operators can set a container-reachable
    base URL via config alone — but paired with the actual bind port, not the
    port hardcoded in the config. An externally-provided NHX_BASE_URL still
    wins.
    """

    def _write_config(self, tmp_path: Path, body: str) -> str:
        path = tmp_path / "config.yaml"
        path.write_text(body, encoding="utf-8")
        return str(path)

    def test_seeds_base_url_host_from_config_file(self, tmp_path: Path):
        config_path = self._write_config(tmp_path, "platform:\n  base_url: http://172.17.0.1:8080\n")
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080, config_path=config_path), env=env)
        assert env["NHX_BASE_URL"] == "http://172.17.0.1:8080"
        # Embedded PDP base URL follows NHX_BASE_URL.
        assert env["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "http://172.17.0.1:8080"

    def test_uses_actual_bind_port_not_config_port(self, tmp_path: Path):
        # The config hardcodes :8080 but the platform is launched on 59007
        # (as the e2e harness does). NHX_BASE_URL must carry the real bind port
        # so internal in-process clients can reach the server.
        config_path = self._write_config(tmp_path, "platform:\n  base_url: http://172.17.0.1:8080\n")
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=59007, config_path=config_path), env=env)
        assert env["NHX_BASE_URL"] == "http://172.17.0.1:59007"
        assert env["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "http://172.17.0.1:59007"

    def test_local_config_static_pdp_url_does_not_override_actual_bind_port(self, tmp_path: Path):
        config_path = self._write_config(
            tmp_path,
            """
platform:
  base_url: http://0.0.0.0:8080
auth:
  policy_decision_point_base_url: http://localhost:8080
""",
        )
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=59007, config_path=config_path), env=env)
        assert env["NHX_BASE_URL"] == "http://127.0.0.1:59007"
        assert env["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "http://127.0.0.1:59007"

    def test_config_base_url_without_port_gets_bind_port(self, tmp_path: Path):
        config_path = self._write_config(tmp_path, "platform:\n  base_url: http://172.17.0.1\n")
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=9090, config_path=config_path), env=env)
        assert env["NHX_BASE_URL"] == "http://172.17.0.1:9090"

    def test_external_env_still_wins_over_config_file(self, tmp_path: Path):
        config_path = self._write_config(tmp_path, "platform:\n  base_url: http://172.17.0.1:8080\n")
        env: dict[str, str] = {"NHX_BASE_URL": "http://nemo-helix-api:8080"}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080, config_path=config_path), env=env)
        assert env["NHX_BASE_URL"] == "http://nemo-helix-api:8080"

    def test_falls_back_to_bind_derived_when_config_omits_base_url(self, tmp_path: Path):
        config_path = self._write_config(tmp_path, "platform:\n  runtime: docker\n")
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080, config_path=config_path), env=env)
        assert env["NHX_BASE_URL"] == "http://127.0.0.1:8080"

    def test_falls_back_to_bind_derived_when_config_file_missing(self):
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080, config_path="/nonexistent/nhx.yaml"), env=env)
        assert env["NHX_BASE_URL"] == "http://127.0.0.1:8080"

    def test_preserves_https_scheme_from_config(self, tmp_path: Path):
        # A configured https:// base URL must not be downgraded to http://.
        config_path = self._write_config(tmp_path, "platform:\n  base_url: https://172.17.0.1:8080\n")
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=9090, config_path=config_path), env=env)
        assert env["NHX_BASE_URL"] == "https://172.17.0.1:9090"
        assert env["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "https://172.17.0.1:9090"

    def test_malformed_ipv6_config_falls_back_to_bind_derived(self, tmp_path: Path):
        # An unterminated bracketed IPv6 makes urlparse raise ValueError; that
        # must fail soft to the bind-derived default rather than abort startup.
        config_path = self._write_config(tmp_path, "platform:\n  base_url: http://[::1\n")
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080, config_path=config_path), env=env)
        assert env["NHX_BASE_URL"] == "http://127.0.0.1:8080"

    def test_wildcard_host_in_config_normalized_to_loopback(self, tmp_path: Path):
        # The bundled local.yaml sets platform.base_url: http://0.0.0.0:8080.
        # 0.0.0.0 is a bind-only wildcard, so the seeded internal base URL must
        # be normalized to loopback or in-process clients (PDP, readiness) break.
        config_path = self._write_config(tmp_path, "platform:\n  base_url: http://0.0.0.0:8080\n")
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080, config_path=config_path), env=env)
        assert env["NHX_BASE_URL"] == "http://127.0.0.1:8080"

    def test_bundled_default_config_seeds_loopback(self):
        # Regression guard for the real default path: `nemo services run` with no
        # --config uses default_config_path() (bundled local.yaml, which sets
        # http://0.0.0.0:8080). The seeded NHX_BASE_URL must be connectable.
        env: dict[str, str] = {}
        apply_run_environment(_make_config(host="0.0.0.0", port=8080, config_path=default_config_path()), env=env)
        assert env["NHX_BASE_URL"] == "http://127.0.0.1:8080"
