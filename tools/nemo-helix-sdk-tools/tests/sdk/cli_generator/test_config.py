# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from nemo_helix_sdk_tools.sdk.cli_generator.config import CLIConfig, discover_entity_types


class TestShouldSkip:
    """Tests for CLIConfig.should_skip method."""

    def _create_config(self, config_yaml: str) -> CLIConfig:
        """Helper to create a CLIConfig from YAML string."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            return CLIConfig(Path(f.name))

    def test_skip_exact_match(self):
        """Should skip resource when exact path is marked with skip: true."""
        config = self._create_config("""
config:
  - resource: [evaluation]
    skip: true
""")
        assert config.should_skip(["evaluation"]) is True

    def test_skip_subresource_when_parent_skipped(self):
        """Should skip subresources when parent resource is marked with skip: true."""
        config = self._create_config("""
config:
  - resource: [evaluation]
    skip: true
""")
        assert config.should_skip(["evaluation", "benchmarks"]) is True
        assert config.should_skip(["evaluation", "configs"]) is True
        assert config.should_skip(["evaluation", "jobs", "logs"]) is True

    def test_no_skip_unrelated_resource(self):
        """Should not skip unrelated resources."""
        config = self._create_config("""
config:
  - resource: [evaluation]
    skip: true
""")
        assert config.should_skip(["customization"]) is False
        assert config.should_skip(["customization", "configs"]) is False

    def test_no_skip_when_not_configured(self):
        """Should not skip resources that have no config entry."""
        config = self._create_config("""
config:
  - resource: [other]
    methods:
      list:
        columns:
          - id
""")
        assert config.should_skip(["evaluation"]) is False
        assert config.should_skip(["evaluation", "benchmarks"]) is False

    def test_skip_deeply_nested_subresource(self):
        """Should skip deeply nested subresources when any ancestor is skipped."""
        config = self._create_config("""
config:
  - resource: [a]
    skip: true
""")
        assert config.should_skip(["a", "b", "c", "d"]) is True

    def test_skip_false_does_not_skip(self):
        """Should not skip when skip is explicitly false."""
        config = self._create_config("""
config:
  - resource: [evaluation]
    skip: false
""")
        assert config.should_skip(["evaluation"]) is False
        assert config.should_skip(["evaluation", "benchmarks"]) is False

    def test_empty_config(self):
        """Should not skip anything with empty config."""
        config = self._create_config("""
config: []
""")
        assert config.should_skip(["anything"]) is False


class TestShouldSkipMethod:
    """Tests for CLIConfig.should_skip_method method."""

    def _create_config(self, config_yaml: str) -> CLIConfig:
        """Helper to create a CLIConfig from YAML string."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            return CLIConfig(Path(f.name))

    def test_skip_method_when_marked(self):
        """Should skip method when skip: true is set."""
        config = self._create_config("""
config:
  - resource: [filesets]
    methods:
      upload_files:
        skip: true
""")
        assert config.should_skip_method(["filesets"], "upload_files") is True

    def test_no_skip_method_when_not_marked(self):
        """Should not skip method when skip is not set."""
        config = self._create_config("""
config:
  - resource: [filesets]
    methods:
      list:
        columns:
          - name
""")
        assert config.should_skip_method(["filesets"], "list") is False

    def test_no_skip_method_when_not_configured(self):
        """Should not skip method when it has no config entry."""
        config = self._create_config("""
config:
  - resource: [filesets]
    methods:
      list:
        columns:
          - name
""")
        assert config.should_skip_method(["filesets"], "upload_files") is False

    def test_skip_method_false_does_not_skip(self):
        """Should not skip when skip is explicitly false."""
        config = self._create_config("""
config:
  - resource: [filesets]
    methods:
      upload_files:
        skip: false
""")
        assert config.should_skip_method(["filesets"], "upload_files") is False


class TestGetAdditionalMethods:
    """Tests for CLIConfig.get_additional_methods method."""

    def _create_config(self, config_yaml: str) -> CLIConfig:
        """Helper to create a CLIConfig from YAML string."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            return CLIConfig(Path(f.name))

    def test_get_additional_methods(self):
        """Should return additional methods config."""
        config = self._create_config("""
config:
  - resource: [filesets]
    additional_methods:
      upload:
        override: filesets/upload.py
      download:
        override: filesets/download.py
""")
        methods = config.get_additional_methods(["filesets"])
        assert "upload" in methods
        assert "download" in methods
        assert methods["upload"]["override"] == "filesets/upload.py"

    def test_get_additional_methods_empty(self):
        """Should return empty dict when no additional methods."""
        config = self._create_config("""
config:
  - resource: [filesets]
    methods:
      list:
        columns:
          - name
""")
        methods = config.get_additional_methods(["filesets"])
        assert methods == {}

    def test_get_additional_methods_nonexistent_resource(self):
        """Should return empty dict for nonexistent resource."""
        config = self._create_config("""
config:
  - resource: [filesets]
    additional_methods:
      upload:
        override: filesets/upload.py
""")
        methods = config.get_additional_methods(["nonexistent"])
        assert methods == {}


class TestGetMethodOverride:
    """Tests for CLIConfig.get_method_override method."""

    def _create_config(self, config_yaml: str) -> CLIConfig:
        """Helper to create a CLIConfig from YAML string."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            return CLIConfig(Path(f.name))

    def test_get_method_override(self):
        """Should return override path when configured."""
        config = self._create_config("""
config:
  - resource: [filesets]
    methods:
      upload_files:
        override: filesets/upload_files.py
""")
        override = config.get_method_override(["filesets"], "upload_files")
        assert override is not None
        assert "filesets/upload_files.py" in str(override)

    def test_get_method_override_none(self):
        """Should return None when no override configured."""
        config = self._create_config("""
config:
  - resource: [filesets]
    methods:
      list:
        columns:
          - name
""")
        assert config.get_method_override(["filesets"], "list") is None

    def test_get_method_override_nonexistent_method(self):
        """Should return None for nonexistent method."""
        config = self._create_config("""
config:
  - resource: [filesets]
    methods:
      list:
        override: filesets/list.py
""")
        assert config.get_method_override(["filesets"], "nonexistent") is None


class TestGetParamHelp:
    """Tests for CLIConfig.get_param_help method."""

    def _create_config(self, config_yaml: str) -> CLIConfig:
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            return CLIConfig(Path(f.name))

    def test_returns_empty_when_not_configured(self):
        config = self._create_config("""
config:
  - resource: [entities]
    methods:
      list:
        columns:
          - name
""")
        assert config.get_param_help(["entities"], "list") == {}

    def test_returns_empty_for_unknown_resource(self):
        config = self._create_config("config: []")
        assert config.get_param_help(["entities"], "list") == {}

    def test_returns_configured_help_text(self):
        config = self._create_config("""
config:
  - resource: [entities]
    methods:
      list:
        param_help:
          entity_type: "The entity type to query"
""")
        result = config.get_param_help(["entities"], "list")
        assert result == {"entity_type": "The entity type to query"}

    def test_resolves_entity_types_placeholder(self, tmp_path):
        (tmp_path / "services").mkdir()
        (tmp_path / "services" / "svc.py").write_text(
            '    __entity_type__: ClassVar[str] = "widget"\n    __entity_type__: ClassVar[str] = "gadget"\n'
        )
        (tmp_path / "packages").mkdir()
        (tmp_path / "plugins").mkdir()

        config = self._create_config("""
config:
  - resource: [entities]
    methods:
      list:
        param_help:
          entity_type: "Known types: {{ENTITY_TYPES}}"
""")

        # Patch get_project_dir to return our tmp_path
        import nemo_helix_sdk_tools.sdk.cli_generator.config as config_mod

        original = config_mod.get_project_dir
        try:
            config_mod.get_project_dir = lambda: tmp_path
            result = config.get_param_help(["entities"], "list")
        finally:
            config_mod.get_project_dir = original

        assert result["entity_type"] == "Known types: gadget, widget"

    def test_placeholder_not_present_skips_scan(self):
        """Help text without placeholder should be returned as-is without scanning."""
        config = self._create_config("""
config:
  - resource: [entities]
    methods:
      list:
        param_help:
          entity_type: "Just a plain help string"
""")
        result = config.get_param_help(["entities"], "list")
        assert result == {"entity_type": "Just a plain help string"}

    def test_multiple_params(self):
        config = self._create_config("""
config:
  - resource: [things]
    methods:
      list:
        param_help:
          thing_type: "The thing type"
          namespace: "Target namespace"
""")
        result = config.get_param_help(["things"], "list")
        assert result == {"thing_type": "The thing type", "namespace": "Target namespace"}


class TestGetWaitConfig:
    """Tests for CLIConfig.get_wait_config method."""

    def _create_config(self, config_yaml: str) -> CLIConfig:
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            return CLIConfig(Path(f.name))

    def test_returns_configured_wait(self):
        config = self._create_config("""
config:
  - resource: [inference, deployments]
    methods:
      create:
        wait:
          type: inference_deployment
          resource_label: deployment
""")
        assert config.get_wait_config(["inference", "deployments"], "create") == {
            "type": "inference_deployment",
            "resource_label": "deployment",
        }

    def test_returns_none_when_not_configured(self):
        config = self._create_config("config: []")

        assert config.get_wait_config(["customization", "jobs"], "create") is None

    def test_rejects_non_mapping_wait_config(self):
        config = self._create_config("""
config:
  - resource: [customization, jobs]
    methods:
      create:
        wait: true
""")
        with pytest.raises(
            ValueError,
            match=r"Invalid wait config for customization\.jobs\.create\. Expected a mapping, got bool\.",
        ):
            config.get_wait_config(["customization", "jobs"], "create")

    def test_rejects_unknown_wait_config_type(self):
        config = self._create_config("""
config:
  - resource: [customization, jobs]
    methods:
      create:
        wait:
          type: unknown
          resource_label: customization job
""")

        with pytest.raises(ValueError, match="Invalid wait config type 'unknown'"):
            config.get_wait_config(["customization", "jobs"], "create")

    def test_returns_platform_job_wait_config(self):
        config = self._create_config("""
config:
  - resource: [customization, jobs]
    methods:
      create:
        wait:
          type: platform_job
          resource_label: customization job
""")

        assert config.get_wait_config(["customization", "jobs"], "create") == {
            "type": "platform_job",
            "resource_label": "customization job",
        }

    def test_rejects_unhashable_wait_config_type(self):
        config = self._create_config("""
config:
  - resource: [customization, jobs]
    methods:
      create:
        wait:
          type: [platform_job]
          resource_label: customization job
""")

        with pytest.raises(
            ValueError,
            match=r"type=\['platform_job'\].*resource_path=\['customization', 'jobs'\].*method_name='create'.*Expected one of:",
        ):
            config.get_wait_config(["customization", "jobs"], "create")

    def test_rejects_missing_wait_resource_label(self):
        config = self._create_config("""
config:
  - resource: [customization, jobs]
    methods:
      create:
        wait:
          type: inference_deployment
""")

        with pytest.raises(ValueError, match="Invalid wait config resource_label None"):
            config.get_wait_config(["customization", "jobs"], "create")

    def test_rejects_empty_wait_resource_label(self):
        config = self._create_config("""
config:
  - resource: [customization, jobs]
    methods:
      create:
        wait:
          type: inference_deployment
          resource_label: " "
""")

        with pytest.raises(ValueError, match="Invalid wait config resource_label ' '"):
            config.get_wait_config(["customization", "jobs"], "create")


class TestGetWatchConfig:
    """Tests for CLIConfig.get_watch_config method."""

    def _create_config(self, config_yaml: str) -> CLIConfig:
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            return CLIConfig(Path(f.name))

    def test_returns_configured_watch(self):
        config = self._create_config("""
config:
  - resource: [customization, jobs]
    methods:
      create:
        watch:
          type: platform_job
          resource_label: customization job
""")
        assert config.get_watch_config(["customization", "jobs"], "create") == {
            "type": "platform_job",
            "resource_label": "customization job",
        }

    def test_returns_inference_deployment_watch_config(self):
        config = self._create_config("""
config:
  - resource: [inference, deployments]
    methods:
      create:
        watch:
          type: inference_deployment
          resource_label: deployment
""")

        assert config.get_watch_config(["inference", "deployments"], "create") == {
            "type": "inference_deployment",
            "resource_label": "deployment",
        }


class TestGetExamples:
    def _create_config(self, config_yaml: str) -> CLIConfig:
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            return CLIConfig(Path(f.name))

    def test_returns_configured_examples(self):
        config = self._create_config("""
config:
  - resource: [intake, ingest, spans]
    methods:
      create:
        examples:
          - nemo intake ingest spans create --input-file spans.json
""")

        assert config.get_examples(["intake", "ingest", "spans"], "create") == [
            "nemo intake ingest spans create --input-file spans.json"
        ]

    def test_rejects_invalid_examples(self):
        config = self._create_config("""
config:
  - resource: [intake, ingest, spans]
    methods:
      create:
        examples: []
""")

        with pytest.raises(ValueError, match="Invalid CLI examples"):
            config.get_examples(["intake", "ingest", "spans"], "create")


class TestDiscoverEntityTypes:
    """Tests for discover_entity_types function."""

    # The regex requires a type annotation between __entity_type__ and =
    # (e.g. __entity_type__: ClassVar[str] = "value"), matching real production usage.
    _ENTITY_TYPE_LINE = '    __entity_type__: ClassVar[str] = "{value}"\n'

    def test_finds_entity_types_in_services(self, tmp_path):
        (tmp_path / "services").mkdir()
        (tmp_path / "packages").mkdir()
        (tmp_path / "plugins").mkdir()
        (tmp_path / "services" / "model.py").write_text(self._ENTITY_TYPE_LINE.format(value="model"))
        (tmp_path / "services" / "adapter.py").write_text(self._ENTITY_TYPE_LINE.format(value="adapter"))

        result = discover_entity_types(tmp_path)
        assert result == ["adapter", "model"]

    def test_finds_entity_types_in_packages_and_plugins(self, tmp_path):
        (tmp_path / "services").mkdir()
        (tmp_path / "packages").mkdir()
        (tmp_path / "plugins").mkdir()
        (tmp_path / "packages" / "pkg.py").write_text(self._ENTITY_TYPE_LINE.format(value="pkg_type"))
        (tmp_path / "plugins" / "plug.py").write_text(self._ENTITY_TYPE_LINE.format(value="plug_type"))

        result = discover_entity_types(tmp_path)
        assert "pkg_type" in result
        assert "plug_type" in result

    def test_deduplicates_across_files(self, tmp_path):
        (tmp_path / "services").mkdir()
        (tmp_path / "packages").mkdir()
        (tmp_path / "plugins").mkdir()
        (tmp_path / "services" / "a.py").write_text(self._ENTITY_TYPE_LINE.format(value="model"))
        (tmp_path / "packages" / "b.py").write_text(self._ENTITY_TYPE_LINE.format(value="model"))

        result = discover_entity_types(tmp_path)
        assert result.count("model") == 1

    def test_returns_sorted(self, tmp_path):
        (tmp_path / "services").mkdir()
        (tmp_path / "packages").mkdir()
        (tmp_path / "plugins").mkdir()
        content = self._ENTITY_TYPE_LINE.format(value="zebra") + self._ENTITY_TYPE_LINE.format(value="apple")
        (tmp_path / "services" / "a.py").write_text(content)

        result = discover_entity_types(tmp_path)
        assert result == sorted(result)

    def test_empty_when_no_matches(self, tmp_path):
        (tmp_path / "services").mkdir()
        (tmp_path / "packages").mkdir()
        (tmp_path / "plugins").mkdir()
        (tmp_path / "services" / "a.py").write_text("x = 1\n")

        assert discover_entity_types(tmp_path) == []

    def test_missing_scan_dir_ignored(self, tmp_path):
        """Should not crash when a scan directory (e.g. plugins/) doesn't exist."""
        (tmp_path / "services").mkdir()
        (tmp_path / "packages").mkdir()
        # plugins/ intentionally absent
        (tmp_path / "services" / "a.py").write_text(self._ENTITY_TYPE_LINE.format(value="thing"))

        result = discover_entity_types(tmp_path)
        assert "thing" in result
