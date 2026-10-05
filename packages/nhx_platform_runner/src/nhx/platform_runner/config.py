# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration resolution for the platform runner."""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse

from nhx.common.config import (
    NHX_CONFIG_FILE_PATH_ENV_VAR,
    NHX_CONTROLLERS_ENV_VAR,
    NHX_SERVICES_ENV_VAR,
    NHX_SIDECARS_ENV_VAR,
    Configuration,
)
from nhx.common.service import Service
from nhx.platform_runner.loader import ControllerRunFunc
from nhx.platform_runner.registry import (
    AVAILABLE_SIDECARS,
    get_available_controllers,
    get_available_services,
    get_controller_groups,
    get_default_controllers,
    get_service_groups,
)

DEFAULT_SCOPE = "default"
DEFAULT_PLATFORM_BIND_HOST = "0.0.0.0"
DEFAULT_LOCAL_SERVICES_BIND_HOST = "127.0.0.1"
DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS = 5

_SCOPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INSTANCES_DIRNAME = "instances"
_SOCKET_FILENAME = "nemo-helix.sock"
_LOG_FILENAME = "services.log"
_QUICKSTART_PLATFORM_CONFIG_FILENAME = "platform-config.yaml"
_SECRETS_ALLOW_KEY_CREATION_ENV_VAR = "NHX_SECRETS_ALLOW_KEY_CREATION"
_SECRETS_LOCAL_KEY_CREATION_PATH_ENV_VAR = "NHX_SECRETS_LOCAL_KEY_CREATION_PATH"
_QUICKSTART_LOCAL_KEY_CREATION_PATH = "/data/nhx-encryption-key.txt"


@dataclass
class HelixAppConfig:
    """Service selection and listener binding for a platform app/server."""

    services: Sequence[str] | None = None
    service_group: str | None = None
    controllers: Sequence[str] | None = None
    controller_group: str | None = None
    sidecars: Sequence[str] | None = None
    config_path: str | None = None
    scope: str = DEFAULT_SCOPE
    host: str = DEFAULT_PLATFORM_BIND_HOST
    port: int = 8080
    socket_path: str | Path | None = None
    state_root: str | Path | None = None
    runtime_root: str | Path | None = None
    log_path: str | Path | None = None
    keep_alive_timeout_seconds: int = DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        self.scope = validate_scope(self.scope)
        self.socket_path = _resolve_socket_path(self.socket_path)
        self.state_root = _resolve_absolute_path(self.state_root, "state root")
        self.runtime_root = _resolve_absolute_path(self.runtime_root, "runtime root")
        self.log_path = _resolve_absolute_path(self.log_path, "log path")
        self.keep_alive_timeout_seconds = validate_keep_alive_timeout_seconds(self.keep_alive_timeout_seconds)

    @property
    def state_root_path(self) -> Path:
        return Path(self.state_root) if self.state_root is not None else default_state_root()

    @property
    def runtime_root_path(self) -> Path:
        return Path(self.runtime_root) if self.runtime_root is not None else default_runtime_root()

    def state_dir(self, *, create: bool = False) -> Path:
        path = self.state_root_path / _INSTANCES_DIRNAME / self.scope
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def runtime_dir(self, *, create: bool = False) -> Path:
        if self.runtime_root is None and self.socket_path is not None:
            path = Path(self.socket_path).parent
        else:
            path = self.runtime_root_path / self.scope
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def socket_file_path(self) -> Path:
        if self.socket_path is not None:
            return Path(self.socket_path)
        return self.runtime_dir() / _SOCKET_FILENAME

    def log_file_path(self, *, create_parent: bool = False) -> Path:
        path = Path(self.log_path) if self.log_path is not None else self.state_dir() / _LOG_FILENAME
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path


def validate_scope(scope: str) -> str:
    """Ensure *scope* is safe to use for local state and socket paths."""
    if not _SCOPE_RE.fullmatch(scope):
        raise ValueError(f"Invalid scope: {scope!r}")
    return scope


def validate_keep_alive_timeout_seconds(timeout_seconds: int) -> int:
    """Ensure Uvicorn keep-alive timeout is a positive whole-second value."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise ValueError("keep_alive_timeout_seconds must be greater than 0")
    if timeout_seconds <= 0:
        raise ValueError("keep_alive_timeout_seconds must be greater than 0")
    return timeout_seconds


def default_state_root() -> Path:
    """Return the local services state root."""
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "nhx"
    return Path.home() / ".local" / "state" / "nhx"


def default_runtime_root() -> Path:
    """Return the local services runtime root for sockets and volatile metadata."""
    return default_state_root() / "run"


_IPV4_LOOPBACK = "127.0.0.1"
_IPV6_LOOPBACK = "::1"
_IPV4_WILDCARDS = frozenset({"0.0.0.0"})
_IPV6_WILDCARDS = frozenset({"::", "[::]"})


@dataclass
class ResolvedRunConfiguration:
    services: set[str]
    controllers: set[str]
    sidecars: set[str]
    host: str
    port: int
    config_path: str
    socket_path: str | None = None
    available_services: dict[str, str | Service] = field(default_factory=dict)
    available_controllers: dict[str, str | ControllerRunFunc] = field(default_factory=dict)
    keep_alive_timeout_seconds: int = DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS


def default_config_path() -> str:
    """Return the bundled local config path."""
    return os.environ.get(
        NHX_CONFIG_FILE_PATH_ENV_VAR,
        str(files("nhx.platform_runner").joinpath("config/local.yaml")),
    )


def resolve_run_configuration(
    config: HelixAppConfig | None = None,
) -> ResolvedRunConfiguration:
    """Resolve and validate platform run configuration.

    Group selectors are convenience shortcuts for callers that are not also
    naming specific services or controllers. Mixing the two is ambiguous, so
    these combinations fail fast instead of silently ignoring the group.
    """
    config = config or HelixAppConfig()
    available_services = get_available_services()
    available_controllers = get_available_controllers()
    available_sidecars = AVAILABLE_SIDECARS
    service_groups = get_service_groups(available_services)
    controller_groups = get_controller_groups(available_controllers)
    default_controllers = set(get_default_controllers(controller_groups))

    selected_services = set(config.services or [])
    selected_controllers = set(config.controllers or [])
    selected_sidecars = set(config.sidecars or [])

    # Explicit selections and group selectors are mutually exclusive. The old
    # entrypoint rejected these combinations, and keeping that behavior avoids a
    # confusing silent-ignore UX for callers.
    if config.service_group and selected_services:
        raise ValueError("--services cannot be combined with --service-group")

    if config.controller_group and selected_controllers:
        raise ValueError("--controllers cannot be combined with --controller-group")

    if config.service_group and not selected_services:
        if config.service_group not in service_groups:
            valid_groups = ", ".join(sorted(service_groups))
            raise ValueError(f"Unknown service group: {config.service_group}. Available groups: {valid_groups}")
        selected_services.update(service_groups[config.service_group])

    if config.controller_group and not selected_controllers:
        if config.controller_group not in controller_groups:
            valid_groups = ", ".join(sorted(controller_groups))
            raise ValueError(f"Unknown controller group: {config.controller_group}. Available groups: {valid_groups}")
        selected_controllers.update(controller_groups[config.controller_group])

    invalid_services = selected_services - set(available_services)
    if invalid_services:
        available = ", ".join(sorted(available_services))
        requested = ", ".join(sorted(invalid_services))
        raise ValueError(f"Unknown services: {requested}. Available services: {available}")

    invalid_controllers = selected_controllers - set(available_controllers)
    if invalid_controllers:
        available = ", ".join(sorted(available_controllers))
        requested = ", ".join(sorted(invalid_controllers))
        raise ValueError(f"Unknown controllers: {requested}. Available controllers: {available}")

    if not selected_services and not selected_controllers and not selected_sidecars:
        # No explicit selection means "run the platform": start the default
        # service group plus the default controller set.
        selected_services.update(service_groups["all"])
        selected_controllers.update(default_controllers)

    invalid_sidecars = selected_sidecars - set(available_sidecars)
    if invalid_sidecars:
        available = ", ".join(sorted(available_sidecars))
        requested = ", ".join(sorted(invalid_sidecars))
        raise ValueError(f"Unknown sidecars: {requested}. Available sidecars: {available}")

    resolved_socket_path = _resolve_socket_path(config.socket_path)

    return ResolvedRunConfiguration(
        services=selected_services,
        controllers=selected_controllers,
        sidecars=selected_sidecars,
        host=config.host,
        port=config.port,
        config_path=config.config_path or default_config_path(),
        socket_path=resolved_socket_path,
        available_services=available_services,
        available_controllers=available_controllers,
        keep_alive_timeout_seconds=config.keep_alive_timeout_seconds,
    )


def _resolve_socket_path(socket_path: str | Path | None) -> str | None:
    if socket_path is None:
        return None
    path = Path(socket_path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"UDS socket path must be absolute: {socket_path}")
    return str(path)


def _resolve_absolute_path(path_value: str | Path | None, label: str) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path_value}")
    return str(path)


def apply_run_environment(
    config: ResolvedRunConfiguration,
    env: MutableMapping[str, str] | None = None,
) -> None:
    """Apply the resolved run configuration to process environment variables.

    Args:
        config: The resolved configuration to apply.
        env: Environment mapping to write to. Defaults to ``os.environ``.
            Accepting an explicit mapping makes this function trivially
            testable without monkeypatching or snapshot fixtures.

    Uses ``setdefault`` for NHX_BASE_URL / NHX_SERVICE_HOST / NHX_SERVICE_PORT
    so that values pre-set by Helm / k8s (deployed mode) are never overwritten.
    In standalone mode these variables are absent, so setdefault fills them in.

    Precedence for NHX_BASE_URL: an externally-provided value (Helm / k8s) wins,
    then the scheme + host of an explicit ``platform.base_url`` from the config
    file combined with the actual bind port, then a value derived entirely from
    the bind host/port. Seeding from the config file lets operators set a base
    URL reachable from inside deployed agent containers (e.g. the Docker bridge
    address) via config alone; without this the bind-derived loopback default
    would silently shadow the configured value.

    Only the scheme and host of the configured ``platform.base_url`` are
    honored — its port is replaced with the port the server actually binds. A
    config that hardcodes ``:8080`` must not point internal clients (and the
    embedded PDP) at 8080 when the platform is launched on another port (which
    the e2e harness always does, and any ``nemo services run --port`` differing
    from 8080 does); doing so leaves internal HTTP clients unable to reach the
    server and the platform never becomes ready. The configured host is run
    through the same wildcard -> loopback normalization as the bind host, so a
    config like ``http://0.0.0.0:8080`` (the bundled ``local.yaml`` default)
    still yields a connectable internal base URL.
    """
    if env is None:
        env = os.environ
    env[NHX_CONFIG_FILE_PATH_ENV_VAR] = config.config_path
    connect_host = _connect_host_for_internal_clients(config.host)
    effective_host = env.setdefault("NHX_SERVICE_HOST", connect_host)
    effective_port = env.setdefault("NHX_SERVICE_PORT", str(config.port))
    if config.socket_path:
        default_base_url = f"unix://{config.socket_path}"
    else:
        config_base_url_parts = _config_file_base_url_parts(config.config_path)
        if config_base_url_parts is not None:
            scheme, config_host = config_base_url_parts
            host_for_url = _bracket_ipv6(_connect_host_for_internal_clients(config_host))
            default_base_url = f"{scheme}://{host_for_url}:{effective_port}"
        else:
            host_for_url = _bracket_ipv6(effective_host)
            default_base_url = f"http://{host_for_url}:{effective_port}"
    base_url = env.setdefault("NHX_BASE_URL", default_base_url)
    # Embedded PDP is usually served from the same platform process, so its
    # self-call origin must stay aligned with the resolved base URL. Deployed
    # mode can still override this explicitly by pre-setting the env var.
    env.setdefault("NHX_AUTH_POLICY_DECISION_POINT_BASE_URL", base_url)
    _apply_quickstart_dev_secrets_defaults(config, env)
    _set_or_clear_env(env, NHX_SERVICES_ENV_VAR, config.services)
    _set_or_clear_env(env, NHX_CONTROLLERS_ENV_VAR, config.controllers)
    _set_or_clear_env(env, NHX_SIDECARS_ENV_VAR, config.sidecars)
    Configuration.clear_cache()


def _set_or_clear_env(env: MutableMapping[str, str], name: str, values: set[str]) -> None:
    if values:
        env[name] = ",".join(sorted(values))
    else:
        env.pop(name, None)


def _apply_quickstart_dev_secrets_defaults(
    config: ResolvedRunConfiguration,
    env: MutableMapping[str, str],
) -> None:
    """Backfill quickstart's dev/demo secrets fallback for legacy configs.

    The secrets service deliberately defaults ``allow_key_creation`` to false.
    Quickstart is the dev/demo exception, and older generated
    ``platform-config.yaml`` files can omit that field while also omitting a
    real encryption provider. Keep the backfill tied to quickstart's config file
    convention so Helm and other named-provider configs continue to fail closed.
    """
    if _SECRETS_ALLOW_KEY_CREATION_ENV_VAR in env:
        return
    if Path(config.config_path).name != _QUICKSTART_PLATFORM_CONFIG_FILENAME:
        return
    try:
        global_settings = Configuration.get_global_settings_from_file(config.config_path)
    except (OSError, ValueError):
        return
    secrets_settings = global_settings.get("secrets")
    if isinstance(secrets_settings, dict):
        if "allow_key_creation" in secrets_settings:
            return
        if _has_explicit_secrets_encryption_provider(secrets_settings.get("encryption")):
            return

    env[_SECRETS_ALLOW_KEY_CREATION_ENV_VAR] = "1"
    env.setdefault(_SECRETS_LOCAL_KEY_CREATION_PATH_ENV_VAR, _QUICKSTART_LOCAL_KEY_CREATION_PATH)


def _has_explicit_secrets_encryption_provider(encryption_settings: object) -> bool:
    if not isinstance(encryption_settings, dict):
        return False
    current_provider = encryption_settings.get("current_provider")
    if isinstance(current_provider, str) and current_provider.strip():
        return True
    providers = encryption_settings.get("providers")
    if not isinstance(providers, dict):
        return False
    return any(isinstance(provider_map, dict) and bool(provider_map) for provider_map in providers.values())


def _config_file_base_url_parts(config_path: str) -> tuple[str, str] | None:
    """Return the (scheme, host) of an explicit ``platform.base_url`` from config.

    Reads the raw YAML rather than the merged config object so a value present
    in the file can be told apart from the schema default (the merged config
    always carries ``base_url``). Returns the URL scheme (defaulting to
    ``"http"`` when the config omits one) and the host component, or ``None``
    when the file is missing, unreadable, does not set ``platform.base_url``, or
    the value has no parseable host.

    The host is returned unbracketed (as ``urlparse`` yields it) so callers can
    normalize it (e.g. wildcard -> loopback) before composing the final URL.
    Only the scheme and host are returned — callers pair them with the actual
    bind port, so a config that hardcodes a port (e.g. ``:8080``) does not point
    internal clients at the wrong port when the platform runs on a different one.
    """
    try:
        global_settings = Configuration.get_global_settings_from_file(config_path)
    except (OSError, ValueError):
        return None
    platform_settings = global_settings.get("platform")
    if not isinstance(platform_settings, dict):
        return None
    base_url = platform_settings.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        return None
    try:
        parsed = urlparse(base_url)
    except ValueError:
        # Malformed value (e.g. an unterminated bracketed IPv6 like
        # ``http://[::1``). Fall back to the bind-derived default rather than
        # aborting startup — a bad config value should fail soft here.
        return None
    if not parsed.hostname:
        return None
    return parsed.scheme or "http", parsed.hostname


def _connect_host_for_internal_clients(host: str) -> str:
    """Translate a bind-address into a connectable address.

    Wildcard addresses (``0.0.0.0``, ``::``) are replaced with the
    corresponding loopback address so that internal HTTP clients (e.g.
    controllers, readiness probes) can actually reach the server.
    """
    stripped = host.strip("[]")
    if stripped in _IPV4_WILDCARDS:
        return _IPV4_LOOPBACK
    if stripped in _IPV6_WILDCARDS:
        return _IPV6_LOOPBACK
    return stripped


def _bracket_ipv6(host: str) -> str:
    """Bracket an IPv6 literal so it can be composed into ``<host>:<port>``.

    Accepts an already-stripped host (no surrounding brackets) and wraps it in
    ``[...]`` when it is an IPv6 literal (contains ``:``); returns other hosts
    unchanged.
    """
    stripped = host.strip("[]")
    return f"[{stripped}]" if ":" in stripped else stripped
