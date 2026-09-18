# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin configuration interface — base class and utilities for plugin-authored config.

Plugin authors subclass :class:`NemoConfig` and declare :attr:`plugin_name` and
:attr:`plugin_description` as ``ClassVar`` strings — mirroring the pattern used by
:class:`~nemo_platform_plugin.service.NemoService`, :class:`~nemo_platform_plugin.controller.NemoController`,
and the other ``_NamedPlugin`` subclasses.

Example::

    from typing import ClassVar
    from pydantic import Field
    from nemo_platform_plugin.config import NemoConfig

    class MyPluginConfig(NemoConfig):
        plugin_name: ClassVar[str] = "my_plugin"
        plugin_description: ClassVar[str] = "Configuration for my plugin."

        log_level: str = Field(default="INFO")
        max_workers: int = Field(default=4)

Load the singleton instance::

    config = MyPluginConfig.get()
    # or equivalently:
    from nemo_platform_plugin.config import get_nemo_config
    config = get_nemo_config(MyPluginConfig)

Override for tests::

    from nemo_platform_plugin.config import set_nemo_config_override, clear_nemo_config_override

    def test_something():
        set_nemo_config_override(MyPluginConfig(log_level="DEBUG"))
        try:
            ...
        finally:
            clear_nemo_config_override(MyPluginConfig)

Environment variable namespace
-------------------------------

The env prefix is derived from ``plugin_name``: ``NEMO_<SAFE_NAME>_`` where
``SAFE_NAME`` is ``plugin_name`` uppercased with non-word characters replaced
by ``_``.

For ``plugin_name = "my_plugin"`` the prefix is ``NEMO_MY_PLUGIN_``.  Nested
fields use ``_`` as the delimiter:

    NEMO_MY_PLUGIN_LOG_LEVEL=DEBUG
    NEMO_MY_PLUGIN_MAX_WORKERS=8

YAML config file
----------------

The config key in ``/etc/nmp/config.yaml`` matches ``plugin_name``:

    my_plugin:
      log_level: DEBUG
      max_workers: 8

The Helm chart value ``platformConfig.my_plugin`` maps directly to this key.
"""

from __future__ import annotations

import logging
import os
import platform
import re
from abc import ABC, abstractmethod
from enum import Enum
from functools import cache
from os import environ
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, Type, TypeVar

import yaml
from nemo_platform_plugin.capabilities import probe_docker
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic._internal._model_construction import ModelMetaclass
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

logger = logging.getLogger(__name__)

NMP_PREFIX_BASE = "NMP_"
NMP_CONFIG_FILE_PATH_ENV_VAR = "NMP_CONFIG_FILE_PATH"
NMP_CONFIG_FILE_PATH_DEFAULT = "/etc/nmp/config.yaml"
NMP_CONFIG_WARNINGS_DISABLED_ENV_VAR = "NMP_CONFIG_WARNINGS_DISABLED"

_NMP_DATA_DIR_ENV_VAR = "NMP_DATA_DIR"
_XDG_DATA_HOME_ENV_VAR = "XDG_DATA_HOME"
_DATA_DIR_NAME = "nemo"
_FALLBACK_DATA_DIR = Path(f"~/.local/share/{_DATA_DIR_NAME}")


def nmp_user_data_dir() -> Path:
    """Return the directory for persistent NeMo Platform local-development state.

    Resolution order:

    1. ``$NMP_DATA_DIR`` if set
    2. ``$XDG_DATA_HOME/nemo`` if set
    3. ``~/.local/share/nemo``

    The directory is not created on call — callers should mkdir when they
    actually need to write.
    """
    override = os.environ.get(_NMP_DATA_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get(_XDG_DATA_HOME_ENV_VAR)
    if xdg:
        return Path(xdg).expanduser() / _DATA_DIR_NAME
    return _FALLBACK_DATA_DIR.expanduser()


if environ.get(NMP_CONFIG_WARNINGS_DISABLED_ENV_VAR, "") != "":
    logger.setLevel(logging.ERROR)


def internal_field(**kwargs: Any) -> Any:
    """Field that is loaded from config/env but excluded from public config documentation."""
    extra = dict(kwargs.get("json_schema_extra") or {})
    extra["exclude_from_docs"] = True
    kwargs["json_schema_extra"] = extra
    return Field(**kwargs)


def get_service_config_prefix(service_name: str) -> str:
    """Returns the environment variable prefix for a given service name."""
    if service_name == "platform":
        return NMP_PREFIX_BASE
    safe_name = re.sub(r"\W+", "_", service_name).upper()
    return f"{NMP_PREFIX_BASE}{safe_name}_"


class EnvironmentFirstSettings(BaseSettings):
    """Base class that prefers environment variables over other settings sources."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            dotenv_settings,
            init_settings,
            file_secret_settings,
        )


class ServiceConfig(ABC, EnvironmentFirstSettings):
    """Base class for service configurations read from the global configuration file."""

    @staticmethod
    @abstractmethod
    def global_settings_key() -> str: ...


_T_config = TypeVar("_T_config", bound=ServiceConfig)
_T_platform_config = TypeVar("_T_platform_config", bound="NemoPlatformConfig")


def create_service_config_class(service_name: str) -> Type[ServiceConfig]:
    """Generates a ServiceConfig class with a specific environment variable prefix."""

    class ServiceConfigSubclass(ServiceConfig):
        model_config = SettingsConfigDict(
            env_prefix=get_service_config_prefix(service_name),
            env_nested_delimiter="_",
            extra="allow",
            populate_by_name=True,
        )

        @staticmethod
        def global_settings_key() -> str:
            return service_name

        @classmethod
        def get(cls):
            return Configuration.get_service_config(cls)

    return ServiceConfigSubclass


def _deep_merge_defaults_with_file(defaults: dict, file_overrides: dict) -> dict:
    """Recursively merge file_overrides onto a copy of defaults."""
    result = defaults.copy()
    for key, file_value in file_overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(file_value, dict):
            result[key] = _deep_merge_defaults_with_file(result[key], file_value)
        else:
            result[key] = file_value
    return result


class Configuration:
    """Singleton config loader from YAML files and environment variables."""

    _overrides: ClassVar[dict[Type[ServiceConfig], ServiceConfig]] = {}
    _platform_config_class: ClassVar[Type["NemoPlatformConfig"] | None] = None

    @classmethod
    def set_override(cls, config: ServiceConfig) -> None:
        cls._overrides[type(config)] = config

    @classmethod
    def set_overrides(cls, configs: dict[Type[ServiceConfig], ServiceConfig]) -> None:
        cls._overrides.update(configs)

    @classmethod
    def clear_overrides(cls) -> None:
        cls._overrides.clear()

    @classmethod
    def clear_override(cls, config_type: Type[ServiceConfig]) -> None:
        cls._overrides.pop(config_type, None)

    @classmethod
    def clear_cache(cls) -> None:
        cls._get_cached_config.cache_clear()

    @staticmethod
    def global_settings_to_service_config(global_settings: dict, service_config: Type[_T_config]) -> _T_config:
        key = service_config.global_settings_key()
        if key not in global_settings:
            logger.debug(
                f"Settings for service '{key}' not found in global settings, using default values for '{key}' service."
            )
            return service_config()

        file_settings = global_settings.get(key)
        if not isinstance(file_settings, dict):
            raise ValueError(f"Settings for service '{key}' must be a mapping.")
        default_instance = service_config()
        default_dict = default_instance.model_dump()
        merged = _deep_merge_defaults_with_file(default_dict, file_settings)
        return service_config(**merged)

    @staticmethod
    def get_global_settings_from_file(yaml_file_path: str) -> dict:
        if not Path(yaml_file_path).is_file():
            logger.debug("Configuration file not found, using defaults", extra={"path": yaml_file_path})
            return {}
        with open(yaml_file_path, "r") as file:
            try:
                global_settings = yaml.safe_load(file)
                if not isinstance(global_settings, dict):
                    raise ValueError(
                        f"NeMo Platform configuration file '{yaml_file_path}' does not contain a top-level dictionary."
                    )
                if not global_settings:
                    raise ValueError(
                        "NeMo Platform configuration file is empty. Please check the configuration file path, "
                        + f"or unset the environment variable '{NMP_CONFIG_FILE_PATH_ENV_VAR}' to use default values."
                    )
                return global_settings
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing NeMo Platform configuration file as YAML: {e}") from e

    @staticmethod
    def get_global_settings_from_env() -> dict:
        yaml_file_path = environ.get(NMP_CONFIG_FILE_PATH_ENV_VAR)
        if not yaml_file_path:
            yaml_file_path = NMP_CONFIG_FILE_PATH_DEFAULT
        return Configuration.get_global_settings_from_file(yaml_file_path)

    @staticmethod
    @cache
    def _get_cached_config(service_config: Type[_T_config]) -> _T_config:
        return Configuration.global_settings_to_service_config(
            Configuration.get_global_settings_from_env(), service_config
        )

    @classmethod
    def get_service_config(cls, service_config: Type[_T_config]) -> _T_config:
        if service_config in cls._overrides:
            override = cls._overrides[service_config]
            if not isinstance(override, service_config):
                raise TypeError(f"Config override for {service_config.__name__} has type {type(override).__name__}")
            return override
        return cls._get_cached_config(service_config)

    @classmethod
    def register_platform_config_class(cls, platform_config_class: Type[_T_platform_config]) -> None:
        """Register the config class returned by get_platform_config().

        Platform packages can use this to provide an extended platform config
        without monkey-patching Configuration.
        """
        if not issubclass(platform_config_class, NemoPlatformConfig):
            raise TypeError("platform_config_class must be a NemoPlatformConfig subclass")
        if platform_config_class.global_settings_key() != "platform":
            raise ValueError("platform_config_class must use global settings key 'platform'")
        cls._platform_config_class = platform_config_class
        cls.clear_cache()

    @classmethod
    def get_platform_config_class(cls) -> Type["NemoPlatformConfig"]:
        """Return the currently registered platform config class."""
        if cls._platform_config_class is None:
            raise RuntimeError("No platform config class is registered.")
        return cls._platform_config_class

    @classmethod
    def get_platform_config(cls) -> "NemoPlatformConfig":
        return cls.get_service_config(cls.get_platform_config_class())

    @staticmethod
    def get_service_config_from_file(filename: str, service_config: Type[_T_config]) -> _T_config:
        return Configuration.global_settings_to_service_config(
            Configuration.get_global_settings_from_file(filename), service_config
        )


def get_service_config(service_config: Type[_T_config]) -> _T_config:
    """Convenience function for Configuration.get_service_config."""
    return Configuration.get_service_config(service_config)


def register_platform_config_class(platform_config_class: Type[_T_platform_config]) -> None:
    """Register the config class returned by get_platform_config()."""
    Configuration.register_platform_config_class(platform_config_class)


def get_platform_config_class() -> Type[NemoPlatformConfig]:
    """Return the currently registered platform config class."""
    return Configuration.get_platform_config_class()


# ---------------------------------------------------------------------------
# Platform configuration types
# ---------------------------------------------------------------------------

# Regex for env vars that set per-service URLs (e.g. NMP_FILES_URL).
_PLATFORM_SERVICE_URL_ENV_PATTERN = re.compile(r"^NMP_([A-Z0-9]+)_URL$")

# Addresses that refer to localhost/loopback interfaces
LOOPBACK_ADDRESSES = ("localhost", "0.0.0.0", "::1", "127.0.0.1")


def _is_running_in_docker() -> bool:
    """Detect if the current process is running inside a Docker container."""
    if Path("/.dockerenv").exists():
        return True
    try:
        with open("/proc/self/cgroup", "r") as f:
            return any("docker" in line or "containerd" in line for line in f)
    except (FileNotFoundError, PermissionError):
        return False


def determine_loopback_override() -> str | None:
    """Automatically determine the appropriate loopback address override for jobs."""
    system = platform.system()
    running_in_docker = _is_running_in_docker()

    if system == "Darwin":
        logger.debug("Detected macOS: using host.docker.internal for Docker jobs")
        return "host.docker.internal"

    if running_in_docker:
        import socket

        container_hostname = socket.gethostname()
        logger.debug(f"Detected Docker container: using container hostname {container_hostname}")
        return container_hostname

    logger.debug("No loopback override needed (likely Linux host network)")
    return None


def validate_docker_available() -> bool:
    """Validate Docker reachability with an uncached probe (≤5s).

    Thin wrapper over :func:`nemo_platform_plugin.capabilities.probe_docker`.
    Prefer ``probe_docker`` when callers need the failure detail or a
    ``docker_host`` override.

    Uses ``use_cache=False`` so :meth:`NemoPlatformConfig.validate_runtime`
    soft-downgrade does not pin a Docker-unavailable verdict into the
    process-wide cache before jobs/deployments registry construction.
    Long-lived server paths that want memoization should call
    ``probe_docker()`` directly.
    """
    return probe_docker(use_cache=False).available


class ImagePullSecret(BaseSettings):
    """Kubernetes image pull secret reference."""

    name: str = Field(description="Kubernetes Secret name for pulling images")


class DockerConfig(BaseModel):
    """Shared Docker configuration for services using Docker backends.

    This configuration is shared between services like jobs and models that
    need to manage Docker containers with GPU resources.

    The reserved_gpu_device_ids field controls which GPUs are available:
    - "all" (default): Auto-detect and use all available GPUs
    - "none" or "": Explicitly disable GPU support
    - Comma-separated list: Use specific GPU device IDs (e.g., "0,1,2,3")
    """

    reserved_gpu_device_ids: str = Field(
        default="all",
        description="GPU device IDs to reserve for the Docker GPU pool. "
        "Use 'all' to auto-detect and use all available GPUs, 'none' or empty string "
        "to disable GPU support, or a comma-separated list of device IDs (e.g., '0,1,2,3').",
    )

    @field_validator("reserved_gpu_device_ids", mode="before")
    @classmethod
    def parse_reserved_gpu_device_ids(cls, v: Any) -> str:
        if isinstance(v, list):
            raise ValueError(
                "reserved_gpu_device_ids must be a string. Use 'all' for auto-detection, "
                "'none' to disable GPUs, or a comma-separated list like '0,1,2'. "
                f"Got list type: {v}"
            )
        if not isinstance(v, str):
            raise ValueError(f"reserved_gpu_device_ids must be a string, got {type(v).__name__}")
        v = v.strip()
        if v.lower() == "all":
            return "all"
        if v.lower() == "none":
            return "none"
        if not v:
            return ""
        try:
            parts = [p.strip() for p in v.split(",")]
            for part in parts:
                if part:
                    int(part)
        except ValueError as e:
            raise ValueError(
                f"reserved_gpu_device_ids must be 'all', 'none', or comma-separated integers, got: {v}"
            ) from e
        return v

    def get_reserved_gpu_ids(self) -> list[int] | None:
        if self.reserved_gpu_device_ids.lower() == "all":
            return None
        if self.reserved_gpu_device_ids.lower() == "none" or not self.reserved_gpu_device_ids:
            return []
        parts = [p.strip() for p in self.reserved_gpu_device_ids.split(",")]
        return [int(p) for p in parts if p]


class Runtime(str, Enum):
    """Deployment runtime used by the platform (e.g. Kubernetes or Docker)."""

    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> "Runtime":
        if value.lower() == "kubernetes":
            return cls.KUBERNETES
        if value.lower() == "docker":
            return cls.DOCKER
        return cls.NONE


class NemoPlatformConfig(ServiceConfig):
    """Platform-wide configuration settings. It inherits from ServiceConfig and provides Platform-centric settings, which may
    be used by other microservices to interact with other Platform services.

    Environment variables NMP_<SERVICE>_URL (e.g. NMP_FILES_URL) are read and merged into
    service_discovery with the service name lowercased; NMP_BASE_URL sets base_url and is not added to
    service_discovery.
    """

    model_config = SettingsConfigDict(
        env_prefix=get_service_config_prefix("platform"),
        env_nested_delimiter="_",
        extra="allow",
        populate_by_name=True,
    )

    @staticmethod
    def global_settings_key() -> str:
        return "platform"

    @classmethod
    def get(cls) -> NemoPlatformConfig:
        """Return the active platform config.

        ``NemoPlatformConfig`` itself is a public handle for the registered
        platform config class. Platform packages may register a subclass with
        extra behavior, so callers of the base class should receive that active
        subclass. Calls on a concrete subclass still load that subclass directly.
        """
        if cls is NemoPlatformConfig:
            return Configuration.get_platform_config()
        return Configuration.get_service_config(cls)

    services: str = internal_field(
        default="",
        description="Comma-separated list of services to run in this process. If not set, all services will be run. This field is only meant to be set by the deployer.",
    )
    controllers: str = internal_field(
        default="",
        description="Comma-separated list of controllers to run in this process. If not set, no controllers will be run.",
    )
    sidecars: str = internal_field(
        default="",
        description="Comma-separated list of sidecars to run in this process. If not set, no sidecars will be run.",
    )

    runtime: Runtime = Field(
        default=Runtime.DOCKER,
        description="Runtime used by the platform. Used to auto-detect default backends to use.",
    )
    base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for the NeMo Platform api. Used as the default URL for all services.",
    )

    service_discovery: dict[str, str] = internal_field(
        default_factory=dict,
        description=(
            "Map of service names to their URLs. Used to discover services by name (e.g. 'files': 'http://files-service:8080'). "
            "Environment variables NMP_<SERVICE>_URL (e.g. NMP_FILES_URL) are read and merged "
            "into this map with the service name lowercased; NMP_BASE_URL is not added here (it sets base_url)."
        ),
    )

    loopback_address: str | None = Field(
        default=None,
        description=(
            "Optional loopback address override for job containers to reach platform services. "
            "If not specified, automatically determined based on platform: "
            "macOS uses 'host.docker.internal', Docker containers use container hostname, "
            "Linux host network uses no override. Can be set via config file or "
            "NMP_LOOPBACK_ADDRESS env var."
        ),
    )
    image_pull_secrets: list[ImagePullSecret] = Field(
        default_factory=list, description="Global image pull secrets for the platform"
    )
    sandbox_cluster_capable: bool = Field(
        default=False,
        description=(
            "Whether OpenSandbox is installed on this cluster. The Helm chart does not "
            "install OpenSandbox. When false, sandboxed GRPO fails at compile time. "
            "Eval and other services should read this same field."
        ),
    )
    sandbox_server_domain: str | None = Field(
        default=None,
        description=(
            "In-cluster OpenSandbox Service DNS with no scheme, injected as "
            "OPEN_SANDBOX_DOMAIN on Kubernetes/Volcano job pods when "
            "sandbox_cluster_capable is true."
        ),
    )
    sandbox_server_protocol: str | None = Field(
        default=None,
        description=(
            "Scheme jobs use to reach OpenSandbox (http or https). In-cluster Services "
            "speak http. RL compiles this into Gym host_provider_options; leave unset "
            "and NeMo-RL health URLs default to https."
        ),
    )
    sandbox_api_key_secret: str | None = Field(
        default=None,
        description=(
            "Kubernetes Secret in the job namespace holding the OpenSandbox API key. "
            "Injected as OPEN_SANDBOX_API_KEY via secretKeyRef when sandbox_cluster_capable "
            "is true. Copy the Secret from opensandbox-system into the release namespace."
        ),
    )
    sandbox_api_key_secret_key: str = Field(
        default="api-key",
        description="Key inside sandbox_api_key_secret that stores the OpenSandbox API key.",
    )
    image_registry: str = Field(
        default="my-registry",
        description="Docker registry for NeMo Platform images (e.g., 'nvcr.io/nvidia/nemo-platform').",
    )
    image_tag: str = Field(
        default="local",
        description="Default tag for NeMo Platform images.",
    )

    docker: DockerConfig = Field(
        default_factory=DockerConfig,
        description="Shared Docker configuration for services using Docker backends.",
    )
    ngc_api_key_secret: str = Field(
        default="system/ngc-api-key",
        description="Name of the secret containing the default NGC API key. Defaults to 'system/ngc-api-key'.",
    )
    ngc_api_key_env_var: str = Field(
        default="NGC_API_KEY",
        description="Environment variable name to source the NGC API key from.",
    )
    seed_on_startup: bool = internal_field(
        default=False,
        description=(
            "When true, run the platform seed task once in the background after the API server starts. "
            "Use for single nmp-api container deployments where a separate Helm Job is not used. "
            "The task waits for entities, auth, and files to be ready, then runs guardrails/evaluator/data-designer seeding. "
            "Set via config file (platform.seed_on_startup) or NMP_SEED_ON_STARTUP env var."
        ),
    )
    redirect_root_to_studio: bool = Field(
        default=True,
        description="When true, GET / returns a 301 redirect to /studio. When false, GET / returns 404. This is used to redirect the root URL to the Studio UI.",
    )

    def get_services(self) -> list[str]:
        """Get the list of services that are local to this process."""
        return [s.strip() for s in self.services.split(",") if s.strip()]

    def _is_service_local(self, api_name: str) -> bool:
        return api_name in self.get_services()

    def get_service_url(self, api_name: str) -> str:
        """Get the URL for a given api name.

        Checks service_discovery first, then falls back to base_url.
        Subclasses (e.g. in nmp-common) may override to add local-service routing.
        """
        if api_name in self.service_discovery:
            return self.service_discovery[api_name]
        return self.base_url

    def create_service_pattern(self) -> re.Pattern[str] | None:
        return re.compile(r"/apis/([a-z]+(?:-[a-z]+)*)/")

    @model_validator(mode="before")
    @classmethod
    def merge_service_url_env_vars(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        sd = dict(values.get("service_discovery") or {})
        for key, value in environ.items():
            if not value:
                continue
            match = _PLATFORM_SERVICE_URL_ENV_PATTERN.match(key)
            if not match:
                continue
            service_name = match.group(1)
            if service_name == "BASE":
                continue
            sd[service_name.lower()] = value
        values["service_discovery"] = sd
        return values

    @model_validator(mode="before")
    @classmethod
    def handle_docker_env_vars(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        docker_config = values.get("docker", {})
        if isinstance(docker_config, dict) and "reserved_gpu_device_ids" not in docker_config:
            env_var = environ.get("NMP_DOCKER_RESERVED_GPU_DEVICE_IDS")
            if env_var:
                docker_config["reserved_gpu_device_ids"] = env_var
                values["docker"] = docker_config
        return values

    @model_validator(mode="after")
    def validate_ngc_api_key_secret(self) -> Self:
        if not self.ngc_api_key_secret:
            return self
        parts = self.ngc_api_key_secret.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("platform.ngc_api_key_secret must be in the form of 'workspace/name'")

        return self

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        if self.runtime == Runtime.DOCKER:
            if not validate_docker_available():
                # Deprecated convenience: Runtime is topology, not capability.
                # Capability probes (nemo_platform_plugin.capabilities) own Docker
                # availability. Soft-downgrade remains for one release; AIRCORE-972
                # removes or shrinks Runtime.NONE as a Docker-absence signal.
                logger.warning(
                    "Docker is not available, setting runtime to NONE "
                    "(deprecated: prefer capability probes; see AIRCORE-972)"
                )
                self.runtime = Runtime.NONE
        return self

    def to_shared_envvars(self, disable_warnings: bool = True, loopback_address: str | None = None) -> dict[str, str]:
        env_prefix = get_service_config_prefix(self.global_settings_key())
        envvars = {
            f"{env_prefix}BASE_URL": self.base_url,
            f"{env_prefix}JOBS_URL": self.get_service_url("jobs"),
            f"{env_prefix}FILES_URL": self.get_service_url("files"),
            f"{env_prefix}MODELS_URL": self.get_service_url("models"),
            f"{env_prefix}SECRETS_URL": self.get_service_url("secrets"),
        }
        if disable_warnings:
            envvars[NMP_CONFIG_WARNINGS_DISABLED_ENV_VAR] = "1"
        effective_override = loopback_address or self.loopback_address or determine_loopback_override()
        if effective_override:
            for key in list(envvars.keys()):
                for loopback in LOOPBACK_ADDRESSES:
                    if loopback in envvars[key]:
                        envvars[key] = envvars[key].replace(loopback, effective_override)
                        break
        return envvars


def get_nemo_platform_config() -> NemoPlatformConfig:
    """Returns the platform configuration singleton.

    Delegates to Configuration.get_platform_config() so that subclass overrides
    (e.g. nmp-common's PlatformConfig with local-service routing) are respected.
    """
    return Configuration.get_platform_config()


# Default implementation; platform packages can register an extended class.
Configuration.register_platform_config_class(NemoPlatformConfig)

# Aliases — nmp-common and services use PlatformConfig / get_platform_config
PlatformConfig = NemoPlatformConfig
get_platform_config = get_nemo_platform_config


class CommonServiceConfig(create_service_config_class("service")):  # ty: ignore[unsupported-base]
    """Common configuration shared by all services.

    Reads from env vars with prefix ``NMP_SERVICE_`` and YAML key ``service``.
    """

    log_format: Literal["json", "plain"] = Field(
        default="plain", alias="LOG_FORMAT", description="Format for logs generated by the service"
    )
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = Field(
        "INFO", alias="LOG_LEVEL", description="Logging level for the NeMo Platform."
    )
    scheme: str = Field(default="http", description="Scheme for the NeMo Platform service.")
    host: str = Field(default="127.0.0.1", description="Host for the NeMo Platform service.")
    port: int = Field(default=8080, description="Port for the NeMo Platform service.")

    def get_host_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


def get_common_service_config() -> CommonServiceConfig:
    return Configuration.get_service_config(CommonServiceConfig)


__all__ = [
    "CommonServiceConfig",
    "Configuration",
    "DockerConfig",
    "EnvironmentFirstSettings",
    "ImagePullSecret",
    "NemoConfig",
    "NemoPlatformConfig",
    "PlatformConfig",
    "Runtime",
    "ServiceConfig",
    "clear_nemo_config_override",
    "clear_nemo_config_overrides",
    "create_service_config_class",
    "get_common_service_config",
    "get_nemo_config",
    "get_nemo_platform_config",
    "get_platform_config",
    "get_platform_config_class",
    "get_service_config",
    "get_service_config_prefix",
    "internal_field",
    "nmp_user_data_dir",
    "register_platform_config_class",
    "set_nemo_config_override",
]

# ---------------------------------------------------------------------------
# Internal TypeVar (not exported)
# ---------------------------------------------------------------------------

_T = TypeVar("_T", bound="NemoConfig")

_NEMO_ENV_PREFIX = "NEMO_"


def _get_plugin_env_prefix(plugin_name: str) -> str:
    """Return the env-var prefix for a plugin: ``NEMO_<SAFE_NAME>_``."""
    safe_name = re.sub(r"\W+", "_", plugin_name).upper()
    return f"{_NEMO_ENV_PREFIX}{safe_name}_"


# ---------------------------------------------------------------------------
# Test override utilities — thin wrappers around Configuration
# ---------------------------------------------------------------------------


def get_nemo_config(cls: type[_T]) -> _T:
    """Return the singleton config instance for *cls*.

    Equivalent to ``cls.get()``.  Provided as a standalone function so
    callers can import a single name and use it for all config types:

        from nemo_platform_plugin.config import get_nemo_config
        config = get_nemo_config(MyPluginConfig)
    """
    return Configuration.get_service_config(cls)


def set_nemo_config_override(config: NemoConfig) -> None:
    """Override config for tests.  The override bypasses file/env loading entirely.

    Call :func:`clear_nemo_config_override` (or :func:`clear_nemo_config_overrides`)
    after the test to avoid polluting other tests.

        set_nemo_config_override(MyPluginConfig(log_level="DEBUG"))
    """
    Configuration.set_override(config)


def clear_nemo_config_override(cls: type[NemoConfig]) -> None:
    """Remove the test override for a specific config class."""
    Configuration.clear_override(cls)


def clear_nemo_config_overrides() -> None:
    """Remove ALL test overrides (across all config types)."""
    Configuration.clear_overrides()


# ---------------------------------------------------------------------------
# Custom metaclass — sets env prefix before Pydantic builds the core schema
# ---------------------------------------------------------------------------

_NEMO_CONFIG_BASE_MARKER = "__is_nemo_config_base__"


class _NemoConfigMeta(ModelMetaclass):
    """Metaclass for :class:`NemoConfig`.

    Intercepts class creation for concrete subclasses and sets the correct
    ``model_config`` (env prefix) **before** Pydantic's ``ModelMetaclass``
    builds the core validation schema.  This is necessary because modifying
    ``model_config`` after the core schema is built (e.g. in
    ``__init_subclass__``) leaves the schema in an inconsistent state where
    declared fields are not accessible on instances.

    Also validates that concrete subclasses declare non-empty ``plugin_name``
    and ``plugin_description`` ClassVars in their own class body.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        # Only apply enforcement to concrete subclasses, not to NemoConfig itself
        # (identified by the absence of the base marker from any parent).
        is_nemo_subclass = any(getattr(b, _NEMO_CONFIG_BASE_MARKER, False) for b in bases)

        if is_nemo_subclass:
            plugin_name: str | None = namespace.get("plugin_name")
            plugin_description: str | None = namespace.get("plugin_description")

            # Exempt abstract intermediates: neither ClassVar declared in own body.
            is_abstract_intermediate = (plugin_name is None) and (plugin_description is None)

            if not is_abstract_intermediate:
                # Description present but name missing → likely a typo.
                if plugin_description is not None and plugin_name is None:
                    raise TypeError(
                        f"'{name}' declares plugin_description but is missing plugin_name: ClassVar[str] = '...'."
                    )

                if plugin_name is not None:
                    if not isinstance(plugin_name, str) or not plugin_name.strip():
                        raise TypeError(f"'{name}'.plugin_name must be a non-empty string.")

                    if not isinstance(plugin_description, str) or not plugin_description.strip():
                        raise TypeError(
                            f"'{name}'.plugin_description must be a non-empty string. "
                            f"Provide a human-readable description of what this plugin's "
                            f"config governs."
                        )

                    # *** KEY: set model_config with the correct env prefix BEFORE
                    # ModelMetaclass.super().__new__ builds the Pydantic core schema.
                    # This ensures the schema is built with the right prefix from the
                    # start, avoiding the stale-schema issue that arises when
                    # model_config is mutated after the fact. ***
                    prefix = _get_plugin_env_prefix(plugin_name)
                    namespace["model_config"] = SettingsConfigDict(
                        env_prefix=prefix,
                        env_nested_delimiter="_",
                        extra="allow",
                        populate_by_name=True,
                    )

                    # Wire global_settings_key before the class is finalised.
                    _name = plugin_name
                    namespace["global_settings_key"] = staticmethod(lambda: _name)

                    logger.debug(
                        "NemoConfig subclass %r: env_prefix=%r yaml_key=%r",
                        name,
                        prefix,
                        plugin_name,
                    )

        return super().__new__(mcs, name, bases, namespace, **kwargs)


# ---------------------------------------------------------------------------
# NemoConfig base class
# ---------------------------------------------------------------------------


class NemoConfig(ServiceConfig, metaclass=_NemoConfigMeta):
    """Base class for plugin configuration.

    Subclasses declare :attr:`plugin_name` and :attr:`plugin_description` as
    ``ClassVar[str]`` — exactly as they would on :class:`~nemo_platform_plugin.service.NemoService`
    or :class:`~nemo_platform_plugin.controller.NemoController`:

    .. code-block:: python

        from typing import ClassVar
        from pydantic import Field
        from nemo_platform_plugin.config import NemoConfig

        class AgentsConfig(NemoConfig):
            plugin_name: ClassVar[str] = "agents"
            plugin_description: ClassVar[str] = "Configuration for the NeMo Platform agents plugin."

            runner_backend: str = Field(default="in_memory")

    The :class:`_NemoConfigMeta` metaclass reads ``plugin_name`` from the class
    body **before** Pydantic builds the core validation schema, so the env prefix
    is correct from the first instantiation without requiring a manual
    ``model_rebuild()``.

    Abstract intermediate base classes (those that declare neither ``plugin_name``
    nor ``plugin_description`` in their own ``__dict__``) are exempt from validation
    and may be used to share common fields across multiple concrete configs.

    Class variables
    ---------------
    .. attribute:: plugin_name
        :type: str

        Unique snake_case name for this plugin's config section.  Must match
        the plugin's :attr:`~nemo_platform_plugin.service.NemoService.name` (and entry-point
        key).  Drives the env prefix (``NEMO_<UPPER>_``) and the YAML key.

    .. attribute:: plugin_description
        :type: str

        Human-readable description of what this configuration governs.
        Required and must be non-empty.
    """

    # Marker read by _NemoConfigMeta to distinguish NemoConfig itself from subclasses.
    __is_nemo_config_base__: ClassVar[bool] = True

    # Default model_config — overridden per concrete subclass by _NemoConfigMeta.
    model_config = SettingsConfigDict(
        env_prefix=_NEMO_ENV_PREFIX,
        env_nested_delimiter="_",
        extra="allow",
        populate_by_name=True,
    )

    @staticmethod
    def global_settings_key() -> str:
        """Return the YAML key for this config section.

        Overridden per-subclass by :class:`_NemoConfigMeta`.  Calling this on the
        bare ``NemoConfig`` base raises ``RuntimeError``.
        """
        raise RuntimeError(
            "NemoConfig.global_settings_key() called on the base class. "
            "Declare plugin_name: ClassVar[str] = '...' on a concrete subclass."
        )

    @classmethod
    def get(cls) -> Self:
        """Return the singleton config instance, loaded from env vars and config file.

        Delegates to :func:`~nmp.common.config.base.Configuration.get_service_config`
        which applies the standard priority order: env vars → config file → defaults.
        The result is cached; call ``Configuration.clear_cache()`` in tests if needed.
        """
        return Configuration.get_service_config(cls)
