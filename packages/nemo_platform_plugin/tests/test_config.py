# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_platform_plugin.config — NemoConfig base class and re-exports."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from nemo_platform_plugin.config import (
    Configuration,
    NemoConfig,
    NemoPlatformConfig,
    PlatformConfig,
    clear_nemo_config_override,
    clear_nemo_config_overrides,
    get_nemo_config,
    get_nemo_platform_config,
    get_platform_config,
    get_platform_config_class,
    register_platform_config_class,
    set_nemo_config_override,
)
from pydantic import Field

# ---------------------------------------------------------------------------
# Helpers — concrete config classes used across tests
# ---------------------------------------------------------------------------


class _WidgetConfig(NemoConfig):
    """Minimal concrete config for testing — not a real plugin."""

    plugin_name: ClassVar[str] = "test_nemo_platform_plugin_widget"
    plugin_description: ClassVar[str] = "Test config for the widget plugin."

    colour: str = Field(default="red")
    count: int = Field(default=1)


class _GadgetConfig(NemoConfig):
    """Second concrete config for isolation tests."""

    plugin_name: ClassVar[str] = "test_nemo_platform_plugin_gadget"
    plugin_description: ClassVar[str] = "Test config for the gadget plugin."

    enabled: bool = Field(default=True)


# ---------------------------------------------------------------------------
# __init_subclass__ enforcement
# ---------------------------------------------------------------------------


def test_valid_subclass_no_error() -> None:
    """A concrete subclass with both ClassVars is accepted at class definition."""

    class _ValidConfig(NemoConfig):
        plugin_name: ClassVar[str] = "test_valid"
        plugin_description: ClassVar[str] = "A valid config."
        some_field: str = "default"


def test_missing_plugin_name_raises() -> None:
    """Concrete subclass without plugin_name raises TypeError at class definition."""
    with pytest.raises(TypeError, match="plugin_name"):

        class _Bad(NemoConfig):
            plugin_description: ClassVar[str] = "Has description but no name."
            some_field: str = "x"


def test_missing_plugin_description_raises() -> None:
    """Concrete subclass without plugin_description raises TypeError."""
    with pytest.raises(TypeError, match="plugin_description"):

        class _Bad(NemoConfig):
            plugin_name: ClassVar[str] = "test_no_desc"
            some_field: str = "x"


def test_empty_plugin_description_raises() -> None:
    """Empty string plugin_description is not accepted."""
    with pytest.raises(TypeError, match="plugin_description"):

        class _Bad(NemoConfig):
            plugin_name: ClassVar[str] = "test_empty_desc"
            plugin_description: ClassVar[str] = ""


def test_empty_plugin_name_raises() -> None:
    """Empty string plugin_name is not accepted."""
    with pytest.raises(TypeError, match="plugin_name"):

        class _Bad(NemoConfig):
            plugin_name: ClassVar[str] = ""
            plugin_description: ClassVar[str] = "Has description."


def test_abstract_intermediate_skips_check() -> None:
    """An intermediate class without plugin_name in its own __dict__ is exempt."""

    class _AbstractBase(NemoConfig):
        # No plugin_name — should not raise
        shared_field: str = "shared"

    # Concrete child must still declare both
    class _Concrete(_AbstractBase):
        plugin_name: ClassVar[str] = "test_abstract_child"
        plugin_description: ClassVar[str] = "Concrete child of abstract base."


def test_abstract_child_missing_name_still_raises() -> None:
    """Concrete child of an abstract intermediate must declare plugin_name."""

    class _AbstractBase(NemoConfig):
        shared_field: str = "shared"

    with pytest.raises(TypeError, match="plugin_name"):

        class _Concrete(_AbstractBase):
            plugin_description: ClassVar[str] = "Missing name."
            some_field: str = "x"


# ---------------------------------------------------------------------------
# env_prefix correctness
# ---------------------------------------------------------------------------


def test_env_prefix_derived_from_plugin_name() -> None:
    """model_config env_prefix matches NEMO_{SAFE_NAME}_."""
    assert _WidgetConfig.model_config.get("env_prefix") == "NEMO_TEST_NEMO_PLATFORM_PLUGIN_WIDGET_"


def test_env_prefix_different_per_subclass() -> None:
    """Each subclass gets its own independent env_prefix."""
    widget_prefix = _WidgetConfig.model_config.get("env_prefix")
    gadget_prefix = _GadgetConfig.model_config.get("env_prefix")
    assert widget_prefix != gadget_prefix
    assert widget_prefix is not None and "WIDGET" in widget_prefix
    assert gadget_prefix is not None and "GADGET" in gadget_prefix


def test_env_nested_delimiter_is_underscore() -> None:
    """Nested fields use _ as delimiter."""
    assert _WidgetConfig.model_config.get("env_nested_delimiter") == "_"


# ---------------------------------------------------------------------------
# global_settings_key
# ---------------------------------------------------------------------------


def test_global_settings_key_returns_plugin_name() -> None:
    """global_settings_key() returns the value of plugin_name."""
    assert _WidgetConfig.global_settings_key() == "test_nemo_platform_plugin_widget"
    assert _GadgetConfig.global_settings_key() == "test_nemo_platform_plugin_gadget"


def test_base_class_global_settings_key_raises() -> None:
    """Calling global_settings_key() on the bare NemoConfig base raises RuntimeError."""
    with pytest.raises(RuntimeError, match="NemoConfig"):
        NemoConfig.global_settings_key()


# ---------------------------------------------------------------------------
# .get() and get_nemo_config() — with test overrides for isolation
# ---------------------------------------------------------------------------


def test_get_returns_default_instance() -> None:
    """cls.get() returns a config instance with Pydantic defaults."""
    clear_nemo_config_overrides()
    config = _WidgetConfig.get()
    assert isinstance(config, _WidgetConfig)
    assert config.colour == "red"
    assert config.count == 1


def test_get_nemo_config_equivalent_to_get() -> None:
    """get_nemo_config(cls) is equivalent to cls.get()."""
    clear_nemo_config_overrides()
    assert type(get_nemo_config(_WidgetConfig)) is _WidgetConfig


def test_set_override_is_returned_by_get() -> None:
    """set_nemo_config_override makes cls.get() return the override."""
    # Use model_validate so ty doesn't flag the metaclass-generated field names.
    override = _WidgetConfig.model_validate({"colour": "blue", "count": 42})
    set_nemo_config_override(override)
    try:
        result = _WidgetConfig.get()
        assert result.colour == "blue"
        assert result.count == 42
    finally:
        clear_nemo_config_override(_WidgetConfig)


def test_clear_override_restores_default() -> None:
    """clear_nemo_config_override removes the override for the specific class."""
    set_nemo_config_override(_WidgetConfig.model_validate({"colour": "green"}))
    clear_nemo_config_override(_WidgetConfig)
    config = _WidgetConfig.get()
    assert config.colour == "red"


def test_overrides_are_independent_per_class() -> None:
    """Overriding one config class does not affect another."""
    set_nemo_config_override(_WidgetConfig.model_validate({"colour": "purple"}))
    try:
        gadget = _GadgetConfig.get()
        assert gadget.enabled is True
    finally:
        clear_nemo_config_override(_WidgetConfig)


def test_clear_all_overrides() -> None:
    """clear_nemo_config_overrides() removes overrides for all classes."""
    set_nemo_config_override(_WidgetConfig.model_validate({"colour": "purple"}))
    set_nemo_config_override(_GadgetConfig.model_validate({"enabled": False}))
    clear_nemo_config_overrides()
    assert _WidgetConfig.get().colour == "red"
    assert _GadgetConfig.get().enabled is True


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_platform_config_is_importable() -> None:
    """PlatformConfig is importable from nemo_platform_plugin.config."""
    assert PlatformConfig is not None
    assert hasattr(PlatformConfig, "model_fields")
    assert "base_url" in PlatformConfig.model_fields


def test_nemo_platform_config_alias() -> None:
    """NemoPlatformConfig is an alias for PlatformConfig."""
    assert NemoPlatformConfig is PlatformConfig


def test_get_platform_config_is_callable() -> None:
    """get_platform_config() is callable and returns a PlatformConfig."""
    result = get_platform_config()
    assert isinstance(result, PlatformConfig)


def test_register_platform_config_class_controls_platform_config_accessors() -> None:
    """A registered platform config class is used by all platform config accessors."""

    class _RegisteredPlatformConfig(NemoPlatformConfig):
        registered_marker: str = "registered"

    previous_platform_config_class = get_platform_config_class()
    register_platform_config_class(_RegisteredPlatformConfig)
    try:
        assert isinstance(Configuration.get_platform_config(), _RegisteredPlatformConfig)
        assert isinstance(get_platform_config(), _RegisteredPlatformConfig)
        assert isinstance(NemoPlatformConfig.get(), _RegisteredPlatformConfig)
        assert isinstance(_RegisteredPlatformConfig.get(), _RegisteredPlatformConfig)
    finally:
        register_platform_config_class(previous_platform_config_class)


def test_get_nemo_platform_config_alias() -> None:
    """get_nemo_platform_config is an alias for get_platform_config."""
    assert get_nemo_platform_config is get_platform_config


def test_validate_docker_available_returns_false_on_connection_failures() -> None:
    from unittest.mock import MagicMock, patch

    from docker.errors import DockerException
    from nemo_platform_plugin.capabilities import reset_capability_cache
    from nemo_platform_plugin.config import validate_docker_available
    from requests.exceptions import ConnectionError as RequestsConnectionError

    for exc in (
        DockerException("boom"),
        RequestsConnectionError("refused"),
        OSError("no such file"),
    ):
        reset_capability_cache()
        with patch("docker.from_env", side_effect=exc):
            assert validate_docker_available() is False

        reset_capability_cache()
        client = MagicMock()
        client.ping.side_effect = exc
        with patch("docker.from_env", return_value=client):
            assert validate_docker_available() is False
        client.close.assert_called()
    reset_capability_cache()


def test_default_client_config_fast_model_falls_back_to_env_default_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_platform_plugin.client.config.config import Config
    from nemo_platform_plugin.client.config.models import ConfigFile

    monkeypatch.setenv("NEMO_DEFAULT_MODEL", "default/quality")
    monkeypatch.delenv("NEMO_FAST_MODEL", raising=False)

    context = Config.create(tmp_path / "config.yaml", ConfigFile()).resolve()

    assert context.default_model == "default/quality"
    assert context.fast_model == "default/quality"
