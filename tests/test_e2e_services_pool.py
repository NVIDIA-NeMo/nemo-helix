# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast

import pytest

import e2e.services_pool as services_pool
import e2e.services_pool_fixtures as services_pool_fixtures


def test_render_e2e_config_for_docker_preserves_container_paths(tmp_path) -> None:
    config = {
        "jobs": {
            "executors": [
                {
                    "provider": "subprocess",
                    "config": {"working_directory": "/data/subprocess-jobs"},
                }
            ]
        },
        "files": {"default_storage_config": {"type": "local", "path": "/data/files"}},
    }

    rendered = services_pool._render_e2e_config_for_backend(config, tmp_path, {"backend": "docker"}, "abc123")

    assert rendered["jobs"]["executors"][0]["config"]["working_directory"] == "/data/subprocess-jobs"
    assert rendered["files"]["default_storage_config"]["path"] == "/data/files"


def test_render_e2e_config_for_subprocess_rewrites_instance_paths(tmp_path) -> None:
    config = {
        "jobs": {
            "executors": [
                {
                    "provider": "subprocess",
                    "config": {"working_directory": ".tmp/e2e/subprocess-jobs"},
                }
            ]
        },
        "files": {"default_storage_config": {"type": "local", "path": ".tmp/e2e/files"}},
    }

    rendered = services_pool._render_e2e_config_for_backend(config, tmp_path, {"backend": "subprocess"}, "abc123")

    assert rendered["jobs"]["executors"][0]["config"]["working_directory"] == str(tmp_path / "subprocess-jobs")
    assert rendered["files"]["default_storage_config"]["path"] == str(tmp_path / "files")


def test_docker_backend_overrides_prefer_e2e_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("IMAGE_REGISTRY", "ghcr.io/example/default")
    monkeypatch.setenv("BAKE_TAG", "default-tag")
    monkeypatch.setenv("NHX_E2E_IMAGE_REGISTRY", "ghcr.io/example/e2e")
    monkeypatch.setenv("NHX_E2E_IMAGE_TAG", "e2e-tag")

    overrides = services_pool._docker_backend_overrides()

    assert overrides == {
        "registry": "ghcr.io/example/e2e",
        "tag": "e2e-tag",
    }


def test_docker_backend_overrides_fall_back_to_ci_bake_env(monkeypatch) -> None:
    monkeypatch.delenv("NHX_E2E_IMAGE_REGISTRY", raising=False)
    monkeypatch.delenv("NHX_E2E_IMAGE_TAG", raising=False)
    monkeypatch.setenv("IMAGE_REGISTRY", "ghcr.io/example/default")
    monkeypatch.setenv("BAKE_TAG", "default-tag")

    overrides = services_pool._docker_backend_overrides()

    assert overrides == {
        "registry": "ghcr.io/example/default",
        "tag": "default-tag",
    }


def test_render_e2e_config_for_docker_compose_preserves_container_paths(tmp_path) -> None:
    config = {
        "jobs": {
            "executors": [
                {
                    "provider": "subprocess",
                    "config": {"working_directory": "/data/subprocess-jobs"},
                }
            ]
        },
        "files": {"default_storage_config": {"type": "local", "path": "/data/files"}},
    }

    rendered = services_pool._render_e2e_config_for_backend(config, tmp_path, {"backend": "docker_compose"}, "abc123")

    assert rendered["jobs"]["executors"][0]["config"]["working_directory"] == "/data/subprocess-jobs"
    assert rendered["files"]["default_storage_config"]["path"] == "/data/files"


def test_start_services_docker_compose_uses_auth_ready_url_override(tmp_path, monkeypatch) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    captured_kwargs = {}

    class FakeDockerComposeBackend:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)
            self.service_url = kwargs["service_url"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def write_logs(self, log_path) -> None:
            log_path.write_text("compose logs\n", encoding="utf-8")

    def fake_wait_for_auth_ready(url, proc) -> bool:
        raise AssertionError("docker compose auth readiness should use DockerComposeE2EBackend.wait_url")

    monkeypatch.setattr(services_pool, "DockerComposeE2EBackend", FakeDockerComposeBackend)
    monkeypatch.setattr(services_pool, "_wait_for_auth_ready", fake_wait_for_auth_ready)

    services = services_pool._start_services_docker_compose(
        config_path,
        {"auth": {"enabled": True}},
        {
            "backend": "docker_compose",
            "compose_file": str(compose_file),
            "service_url": "http://127.0.0.1:38080",
            "auth_ready_url": "${service_url}/health/gateway/ready",
            "wait_url": "http://127.0.0.1:38080/health/ready",
            "lifecycle": "fresh",
        },
        "abc123",
        tmp_path / "services.log",
    )

    assert captured_kwargs["wait_url"] == "http://127.0.0.1:38080/health/gateway/ready"
    assert captured_kwargs["wait_urls"] == ["http://127.0.0.1:38080/health/gateway/ready"]
    assert services.auth_enabled is True
    assert services.url == "http://127.0.0.1:38080"


def test_auth_ready_url_override_uses_url_probe_and_skips_default_probe(monkeypatch) -> None:
    calls = []

    def fake_wait_for_auth_ready_url(url, proc, *, env=None):
        calls.append(("url", url, proc, env))
        return True

    def fake_wait_for_auth_ready(url, proc):
        raise AssertionError("default auth probe should not run when auth_ready_url is configured")

    monkeypatch.setattr(services_pool, "_wait_for_auth_ready_url", fake_wait_for_auth_ready_url)
    monkeypatch.setattr(services_pool, "_wait_for_auth_ready", fake_wait_for_auth_ready)

    assert services_pool._wait_for_configured_auth_ready(
        "http://127.0.0.1:38080",
        None,
        {
            "backend": "subprocess",
            "auth_ready_url": "${service_url}/health/gateway/ready",
            "env": {"NHX_CLIENT_SSL_CERT_FILE": "/tmp/ca.crt"},
        },
    )

    assert calls == [
        (
            "url",
            "http://127.0.0.1:38080/health/gateway/ready",
            None,
            {"NHX_CLIENT_SSL_CERT_FILE": "/tmp/ca.crt"},
        )
    ]


def test_start_services_docker_compose_exposes_log_path_and_captures_logs_on_close(tmp_path, monkeypatch) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    log_path = tmp_path / "services.log"
    events = []

    class FakeDockerComposeBackend:
        def __init__(self, **kwargs) -> None:
            self.service_url = kwargs["service_url"]

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

        def write_logs(self, path) -> None:
            events.append(("logs", path))
            path.write_text("compose logs\n", encoding="utf-8")

    monkeypatch.setattr(services_pool, "DockerComposeE2EBackend", FakeDockerComposeBackend)

    services = services_pool._start_services_docker_compose(
        config_path,
        {},
        {
            "backend": "docker_compose",
            "compose_file": str(compose_file),
            "service_url": "http://127.0.0.1:38080",
            "lifecycle": "fresh",
        },
        "abc123",
        log_path,
    )

    assert services.log_path == log_path
    assert services.close is not None

    services.close()

    assert events == ["start", ("logs", log_path), "stop"]
    assert log_path.read_text(encoding="utf-8") == "compose logs\n"


def test_start_services_docker_compose_resolves_dynamic_port_templates(tmp_path, monkeypatch) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("stale: true\n", encoding="utf-8")
    captured_kwargs = {}

    class FakeDockerComposeBackend:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)
            self.service_url = kwargs["service_url"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def write_logs(self, log_path) -> None:
            log_path.write_text("compose logs\n", encoding="utf-8")

    monkeypatch.setattr(services_pool, "DockerComposeE2EBackend", FakeDockerComposeBackend)
    monkeypatch.setattr(services_pool, "_find_free_port", lambda: 49123)
    monkeypatch.setattr(services_pool.uuid, "uuid4", lambda: SimpleNamespace(hex="deadbeefcafebabe"))

    services = services_pool._start_services_docker_compose(
        config_path,
        {
            "auth": {
                "enabled": True,
                "oidc": {
                    "token_endpoint": "${gateway_url}/application/o/token/",
                },
            },
            "jobs": {
                "executors": [
                    {
                        "config": {
                            "storage": {
                                "additional_volume_mounts": [
                                    {
                                        "volume_name": "authentik-gateway-tls-${gateway_port}",
                                        "mount_path": "/etc/nhx/gateway-tls",
                                    }
                                ]
                            }
                        }
                    }
                ]
            },
        },
        {
            "backend": "docker_compose",
            "compose_file": str(compose_file),
            "compose_project_prefix": "authentik-e2e",
            "dynamic_ports": {
                "gateway": {
                    "host": "127.0.0.1",
                    "scheme": "https",
                }
            },
            "service_url": "${gateway_url}",
            "auth_ready_url": "${gateway_url}/health/gateway/ready",
            "env": {
                "AUTHENTIK_GATEWAY_PORT": "${gateway_port}",
                "AUTHENTIK_GATEWAY_TLS_VOLUME": "authentik-gateway-tls-${gateway_port}",
            },
            "lifecycle": "fresh",
        },
        "abc123",
        tmp_path / "services.log",
    )

    assert captured_kwargs["project_name"] == "authentik-e2e-abc123-deadbeef"
    assert captured_kwargs["service_url"] == "https://127.0.0.1:49123"
    assert captured_kwargs["wait_url"] == "https://127.0.0.1:49123/health/gateway/ready"
    assert captured_kwargs["wait_urls"] == ["https://127.0.0.1:49123/health/gateway/ready"]
    assert "dynamic_ports" not in captured_kwargs
    assert captured_kwargs["env"]["AUTHENTIK_GATEWAY_PORT"] == "49123"
    assert captured_kwargs["env"]["AUTHENTIK_GATEWAY_TLS_VOLUME"] == "authentik-gateway-tls-49123"
    rendered_config = services_pool.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert rendered_config["auth"]["oidc"]["token_endpoint"] == "https://127.0.0.1:49123/application/o/token/"
    assert (
        rendered_config["jobs"]["executors"][0]["config"]["storage"]["additional_volume_mounts"][0]["volume_name"]
        == "authentik-gateway-tls-49123"
    )
    assert services.url == "https://127.0.0.1:49123"
    assert services.compose_project_name == "authentik-e2e-abc123-deadbeef"


def test_start_services_docker_compose_rejects_unfixed_dynamic_ports_with_reuse(tmp_path) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(pytest.UsageError, match="dynamic_ports require explicit ports"):
        services_pool._start_services_docker_compose(
            config_path,
            {},
            {
                "backend": "docker_compose",
                "compose_file": str(compose_file),
                "dynamic_ports": {
                    "gateway": {
                        "host": "127.0.0.1",
                        "scheme": "https",
                    }
                },
                "service_url": "${gateway_url}",
                "lifecycle": "reuse",
            },
            "abc123",
            tmp_path / "services.log",
        )


def test_start_services_docker_compose_allows_fixed_dynamic_ports_with_reuse(tmp_path, monkeypatch) -> None:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("stale: true\n", encoding="utf-8")
    captured_kwargs = {}

    class FakeDockerComposeBackend:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)
            self.service_url = kwargs["service_url"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def write_logs(self, log_path) -> None:
            log_path.write_text("compose logs\n", encoding="utf-8")

    monkeypatch.setattr(services_pool, "DockerComposeE2EBackend", FakeDockerComposeBackend)

    services = services_pool._start_services_docker_compose(
        config_path,
        {
            "auth": {
                "enabled": True,
                "oidc": {
                    "token_endpoint": "${gateway_url}/application/o/token/",
                },
            },
        },
        {
            "backend": "docker_compose",
            "compose_file": str(compose_file),
            "compose_project_name": "authentik-e2e-reuse",
            "dynamic_ports": {
                "gateway": {
                    "host": "127.0.0.1",
                    "port": "18080",
                    "scheme": "https",
                }
            },
            "service_url": "${gateway_url}",
            "auth_ready_url": "${gateway_url}/health/gateway/ready",
            "env": {
                "AUTHENTIK_GATEWAY_PORT": "${gateway_port}",
            },
            "lifecycle": "reuse",
        },
        "abc123",
        tmp_path / "services.log",
    )

    assert captured_kwargs["project_name"] == "authentik-e2e-reuse"
    assert captured_kwargs["service_url"] == "https://127.0.0.1:18080"
    assert captured_kwargs["wait_url"] == "https://127.0.0.1:18080/health/gateway/ready"
    assert captured_kwargs["wait_urls"] == ["https://127.0.0.1:18080/health/gateway/ready"]
    assert captured_kwargs["env"]["AUTHENTIK_GATEWAY_PORT"] == "18080"
    rendered_config = services_pool.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert rendered_config["auth"]["oidc"]["token_endpoint"] == "https://127.0.0.1:18080/application/o/token/"
    assert services.compose_project_name == "authentik-e2e-reuse"


def test_docker_compose_generated_config_uses_repo_cache(tmp_path, tmp_path_factory, monkeypatch) -> None:
    monkeypatch.setattr(services_pool, "_E2E_REPO_ROOT", tmp_path)
    pool = services_pool.E2EServicesPool()
    pool.bind_tmp_path_factory(tmp_path_factory)
    state = services_pool.ModuleConfigState(
        module_id="tests/test_compose.py",
        key=services_pool.ServicesPoolKey(config_hash="composehash"),
        config_path=None,
        config_data={"auth": {"enabled": True}},
        harness_config={
            "backend": "docker_compose",
            "compose_file": "compose.yaml",
            "service_url": "http://127.0.0.1:38080",
            "lifecycle": "fresh",
        },
        config_layers=(),
        auth_enabled=True,
    )

    materialized = pool._materialize_config_path(state)

    assert materialized.config_path is not None
    assert materialized.config_path.is_file()
    assert materialized.config_path.is_relative_to(tmp_path / ".pytest_cache" / "e2e-services")
    assert services_pool.yaml.safe_load(materialized.config_path.read_text(encoding="utf-8")) == {
        "auth": {"enabled": True}
    }


def test_release_for_module_clears_active_binding_while_shared_service_remains(tmp_path) -> None:
    pool = services_pool.E2EServicesPool()
    key = services_pool.ServicesPoolKey(config_hash="shared")
    module_a = cast(pytest.Module, SimpleNamespace(nodeid="tests/test_a.py"))
    module_b = cast(pytest.Module, SimpleNamespace(nodeid="tests/test_b.py"))

    for module in (module_a, module_b):
        pool._module_states[module.nodeid] = services_pool.ModuleConfigState(
            module_id=module.nodeid,
            key=key,
            config_path=None,
            config_data={},
            harness_config={"backend": "subprocess"},
            config_layers=(),
            auth_enabled=False,
        )
        pool._active_service_key_by_module[module.nodeid] = key

    pool._remaining_modules_by_key[key] = {module_a.nodeid, module_b.nodeid}
    pool._running_by_key[key] = services_pool.RunningServices(
        url="http://127.0.0.1:8080",
        log_path=tmp_path / "services.log",
        proc=None,
        config_path=None,
        key=key,
    )

    pool.release_for_module(module_a)

    assert pool.describe_active_module_binding(module_a.nodeid) is None
    module_b_binding = pool.describe_active_module_binding(module_b.nodeid)
    assert module_b_binding is not None
    assert module_b_binding["service_url"] == "http://127.0.0.1:8080"


def test_acquire_for_module_reregisters_released_owner_before_next_release(tmp_path, monkeypatch) -> None:
    pool = services_pool.E2EServicesPool()
    key = services_pool.ServicesPoolKey(config_hash="shared")
    module_a = cast(pytest.Module, SimpleNamespace(nodeid="tests/test_a.py"))
    module_b = cast(pytest.Module, SimpleNamespace(nodeid="tests/test_b.py"))
    terminated = []

    for module in (module_a, module_b):
        pool._module_states[module.nodeid] = services_pool.ModuleConfigState(
            module_id=module.nodeid,
            key=key,
            config_path=tmp_path / "platform.yaml",
            config_data={},
            harness_config={"backend": "subprocess"},
            config_layers=(),
            auth_enabled=False,
        )

    running_services = services_pool.RunningServices(
        url="http://127.0.0.1:8080",
        log_path=tmp_path / "services.log",
        proc=None,
        config_path=tmp_path / "platform.yaml",
        key=key,
    )
    pool._remaining_modules_by_key[key] = {module_a.nodeid, module_b.nodeid}
    pool._running_by_key[key] = running_services
    monkeypatch.setattr(pool, "_terminate_services", terminated.append)

    pool.release_for_module(module_a)
    reacquired = pool.acquire_for_module(module_a)
    pool.release_for_module(module_b)

    assert reacquired is running_services
    assert terminated == []
    module_a_binding = pool.describe_active_module_binding(module_a.nodeid)
    assert module_a_binding is not None
    assert module_a_binding["service_url"] == "http://127.0.0.1:8080"


def test_describe_active_module_binding_ignores_stale_active_key() -> None:
    pool = services_pool.E2EServicesPool()
    key = services_pool.ServicesPoolKey(config_hash="shared")
    module_id = "tests/test_released.py"
    pool._active_service_key_by_module[module_id] = key
    pool._remaining_modules_by_key[key] = {"tests/test_other.py"}
    pool._running_by_key[key] = services_pool.RunningServices(
        url="http://127.0.0.1:8080",
        log_path=None,
        proc=None,
        config_path=None,
        key=key,
    )

    assert pool.describe_active_module_binding(module_id) is None


def test_read_services_log_tail_streams_last_failure_lines(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "services.log"
    lines = [f"line {index}\n" for index in range(services_pool_fixtures._TAIL_LINES_ON_FAILURE + 3)]
    log_path.write_text("".join(lines), encoding="utf-8")

    def fail_read_text(*args, **kwargs) -> str:
        raise AssertionError("services log tail should not materialize the full log with read_text")

    monkeypatch.setattr(type(log_path), "read_text", fail_read_text)

    assert (
        services_pool_fixtures._read_services_log_tail(log_path)
        == lines[-services_pool_fixtures._TAIL_LINES_ON_FAILURE :]
    )
