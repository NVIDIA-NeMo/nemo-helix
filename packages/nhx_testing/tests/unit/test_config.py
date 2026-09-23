# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for E2E config discovery, loading, and merging."""

from pathlib import Path

import pytest
import yaml
from nhx.testing.e2e.config import (
    E2EConfig,
    deep_merge,
    discover_configs,
    infer_backend,
    load_config,
)


class TestDeepMerge:
    def test_merge_simple_scalars(self):
        base = {"key1": "base_value", "key2": 42}
        override = {"key1": "override_value"}
        result = deep_merge(base, override)

        assert result == {"key1": "override_value", "key2": 42}
        assert base == {"key1": "base_value", "key2": 42}  # Verify immutability

    def test_merge_nested_dicts(self):
        base = {
            "service": {"host": "localhost", "port": 8080, "timeout": 30},
            "feature_flags": {"new_ui": False},
        }
        override = {"service": {"port": 9000, "ssl": True}, "feature_flags": {"new_ui": True}}

        assert deep_merge(base, override) == {
            "service": {"host": "localhost", "port": 9000, "timeout": 30, "ssl": True},
            "feature_flags": {"new_ui": True},
        }

    def test_merge_deeply_nested_dicts(self):
        base = {"level1": {"level2": {"level3": {"value": "base"}}}}
        override = {"level1": {"level2": {"level3": {"value": "override", "new_key": "new"}}}}

        assert deep_merge(base, override) == {"level1": {"level2": {"level3": {"value": "override", "new_key": "new"}}}}

    def test_merge_lists_are_replaced(self):
        base = {"items": [1, 2, 3], "tags": ["base"]}
        override = {"items": [4, 5]}

        assert deep_merge(base, override) == {"items": [4, 5], "tags": ["base"]}

    def test_merge_type_replacement(self):
        # Dict can replace scalar and vice versa
        assert deep_merge({"x": "string"}, {"x": {"nested": "value"}}) == {"x": {"nested": "value"}}
        assert deep_merge({"x": {"nested": "value"}}, {"x": "string"}) == {"x": "string"}

    def test_merge_with_empty_dicts(self):
        base = {"key": "value"}
        assert deep_merge(base, {}) == {"key": "value"}
        assert deep_merge({}, base) == {"key": "value"}

    def test_merge_none_values(self):
        base = {"key1": "value", "key2": None}
        override = {"key1": None, "key2": "value"}
        assert deep_merge(base, override) == {"key1": None, "key2": "value"}

    def test_merge_adds_new_keys(self):
        base = {"existing": "value"}
        override = {"new_key": "new_value", "another": {"nested": "data"}}
        assert deep_merge(base, override) == {
            "existing": "value",
            "new_key": "new_value",
            "another": {"nested": "data"},
        }

    def test_merge_immutability(self):
        base = {"nested": {"value": 1}}
        override = {"nested": {"value": 2}}
        result = deep_merge(base, override)

        # Verify inputs weren't modified
        assert base == {"nested": {"value": 1}}
        assert override == {"nested": {"value": 2}}
        # Verify result is independent
        result["nested"]["value"] = 999
        assert base["nested"]["value"] == 1


class TestInferBackend:
    def test_explicit_backend(self):
        assert infer_backend("any_name", {"e2e": {"backend": "docker"}}) == "docker"
        assert infer_backend("any_name", {"e2e": {"backend": "kubernetes"}}) == "kubernetes"

    def test_explicit_backend_invalid(self):
        with pytest.raises(ValueError, match="Invalid backend 'invalid'"):
            infer_backend("any_name", {"e2e": {"backend": "invalid"}})

    def test_infer_from_filename(self):
        # Docker inference
        assert infer_backend("docker", {}) == "docker"
        assert infer_backend("docker_auth_enabled", {}) == "docker"
        assert infer_backend("docker_gpu", {}) == "docker"

        # Kubernetes inference
        assert infer_backend("kubernetes", {}) == "kubernetes"
        assert infer_backend("kubernetes_gpu", {}) == "kubernetes"
        assert infer_backend("k8s", {}) == "kubernetes"
        assert infer_backend("k8s_minikube", {}) == "kubernetes"

        # Default fallback
        assert infer_backend("unknown_config", {}) == "docker"
        assert infer_backend("custom", {}) == "docker"

    def test_explicit_overrides_filename(self):
        assert infer_backend("docker_config", {"e2e": {"backend": "kubernetes"}}) == "kubernetes"


class TestDiscoverConfigs:
    def test_discover_yaml_files(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        (configs_dir / "docker.yaml").write_text("key: value")
        (configs_dir / "docker_auth.yaml").write_text("key: value")
        (configs_dir / "kubernetes.yaml").write_text("key: value")
        (configs_dir / "readme.txt").write_text("ignore me")  # Non-yaml should be ignored

        result = discover_configs(configs_dir)

        assert set(result.keys()) == {"docker", "docker_auth", "kubernetes"}
        assert result["docker"].is_absolute()
        assert result["docker"].name == "docker.yaml"

    def test_discover_nonexistent_directory(self):
        assert discover_configs(Path("/nonexistent")) == {}

    def test_discover_empty_directory(self, tmp_path):
        configs_dir = tmp_path / "empty"
        configs_dir.mkdir()
        assert discover_configs(configs_dir) == {}


class TestLoadConfig:
    def test_load_config_as_is(self, tmp_path):
        config_file = tmp_path / "custom.yaml"
        config_data = {"service": {"port": 8080}, "e2e": {"backend": "docker"}}
        config_file.write_text(yaml.safe_dump(config_data))

        result = load_config(config_file, tmp_path)

        assert result.name == "custom"
        assert result.path == config_file.resolve()
        assert result.backend == "docker"
        assert result.config == config_data

    def test_load_config_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml", tmp_path)

    def test_load_config_empty_yaml(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        assert load_config(config_file, tmp_path).config == {}


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a mock config file for testing."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("test: value")
    return config_file


class TestE2EConfig:
    def test_repr(self, mock_config_file):
        config = E2EConfig(name="docker", path=mock_config_file, backend="docker", config={"key": "value"})
        assert repr(config) == "E2EConfig(name='docker', backend='docker')"


class TestConfigIntegration:
    def test_discover_and_load_workflow(self, tmp_path):
        # Setup configs directory
        configs_dir = tmp_path / "e2e" / "configs"
        configs_dir.mkdir(parents=True)

        # Create test configs (each used as-is)
        (configs_dir / "docker.yaml").write_text(yaml.safe_dump({"override": "docker"}))
        (configs_dir / "docker_gpu.yaml").write_text(yaml.safe_dump({"override": "gpu"}))

        # Discover configs
        discovered = discover_configs(configs_dir)
        assert len(discovered) == 2

        # Load each config (as-is, no inheritance)
        configs = [load_config(path, tmp_path) for path in discovered.values()]

        docker_config = next(c for c in configs if c.name == "docker")
        assert docker_config.config["override"] == "docker"
        gpu_config = next(c for c in configs if c.name == "docker_gpu")
        assert gpu_config.config["override"] == "gpu"
