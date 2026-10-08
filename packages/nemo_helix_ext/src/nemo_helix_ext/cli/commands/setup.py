# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive setup wizard for NeMo Helix.

Full onboarding flow: start local services, register an inference provider,
install AI agent skills, and optionally create a sample workspace and agent.
Supports both interactive and non-interactive (``--auto``) modes.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import httpx
import typer
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.types import CreateSampleAgentRequest, SampleAgentResponse, SampleAgentStreamEvent
from nemo_helix_plugin.capabilities import probe_docker
from nemo_helix_plugin.cli_options import WORKSPACE_HELP
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.entities import DEFAULT_WORKSPACE
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelProviderRequest, UpsertModelProviderRequest
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.common.config import nhx_user_data_dir
from nhx.platform_runner.config import DEFAULT_LOCAL_SERVICES_BIND_HOST, HelixAppConfig, default_state_root
from pydantic import SecretStr
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from nemo_helix_ext.cli.commands.skills import registry as skills_registry
from nemo_helix_ext.cli.commands.skills.agents.custom import CustomPathInstaller
from nemo_helix_ext.cli.commands.skills.base import Scope, Skill
from nemo_helix_ext.cli.commands.skills.installer import BaseAgentInstaller
from nemo_helix_ext.cli.commands.skills.registry import get_installer, list_agent_names, load_skills
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.docker_preflight import DOCKER_PREFLIGHT_MESSAGE, require_docker_for_default_local
from nemo_helix_ext.cli.telemetry import emit
from nemo_helix_ext.cli.telemetry.events import OnboardingStepEvent, TaskStatusEnum
from nemo_helix_ext.client.tls import HttpxTLSConfig, httpx_tls_config_from_env
from nemo_helix_ext.config.config import Config
from nemo_helix_ext.config.models import DEFAULT_BASE_URL, ConfigFile, ConfigParams, LocalServicesConfig, NoAuthUser
from nemo_helix_ext.config.urls import display_url
from nemo_helix_ext.local.install import services_extra_install_command
from nemo_helix_ext.local.process import (
    PortConflict,
    check_port_available_for_start,
    compute_scope,
    format_port_conflict,
    log_path_for,
    start_background,
    stop_instance,
)
from nemo_helix_ext.ui.prompts import (
    UserCancelled,
    is_interactive,
    non_empty_validator,
    prompt_choice,
    prompt_confirm,
    prompt_multiselect,
    prompt_password,
    prompt_search_select,
    prompt_text,
    provider_name_validator,
)

logger = logging.getLogger(__name__)
console = Console(stderr=True)

# Supported Python versions (inclusive).
_SUPPORTED_PYTHON_MIN = (3, 12)
_SUPPORTED_PYTHON_MAX = (3, 14)

CHECK = "[green]✓[/green]"
CROSS = "[red]✗[/red]"
WARN = "[yellow]![/yellow]"


_LEGACY_DIR_NAME = "nm" + "p"


def _legacy_config_dir() -> Path:
    """Return the pre-rename user config directory."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home).expanduser() / _LEGACY_DIR_NAME
    return Path.home() / ".config" / _LEGACY_DIR_NAME


def _legacy_state_dir() -> Path:
    """Return the pre-rename user state directory."""
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / _LEGACY_DIR_NAME
    return Path.home() / ".local" / "state" / _LEGACY_DIR_NAME


def _print_legacy_directory_notice() -> None:
    """Warn setup users when pre-rename config/state directories remain."""
    legacy_dirs = [path for path in (_legacy_config_dir(), _legacy_state_dir()) if path.exists()]
    if not legacy_dirs:
        return

    legacy_list = ", ".join(f"[cyan]{escape(str(path))}[/cyan]" for path in legacy_dirs)
    console.print(
        f"{WARN} Found legacy pre-rename config/state directories from before the {_LEGACY_DIR_NAME} → nhx rename: "
        f"{legacy_list}. NeMo Helix now reads [cyan]{escape(str(Config.get_default_config_path()))}[/cyan] "
        f"and [cyan]{escape(str(default_state_root()))}[/cyan]; the legacy directories are ignored.\n",
        soft_wrap=True,
    )


# ---------------------------------------------------------------------------
# Known provider catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownProvider:
    """A well-known inference provider with pre-configured connection details."""

    name: str
    label: str
    description: str
    host_url: str
    auth_header_format: str | None = None
    default_extra_headers: dict[str, str] | None = None
    env_var: str | None = None
    requires_api_key: bool = True


KNOWN_PROVIDERS: tuple[KnownProvider, ...] = (
    KnownProvider(
        name="nvidia-build",
        label="NVIDIA Build",
        description="NVIDIA-hosted models via build.nvidia.com",
        host_url="https://integrate.api.nvidia.com",
        env_var="NVIDIA_API_KEY",
    ),
    KnownProvider(
        name="openai",
        label="OpenAI",
        description="GPT-4.1, o3, o4-mini",
        host_url="https://api.openai.com/v1",
        env_var="OPENAI_API_KEY",
    ),
    KnownProvider(
        name="anthropic",
        label="Anthropic",
        description="Claude Opus 4, Sonnet 4, Haiku",
        host_url="https://api.anthropic.com",
        auth_header_format="X-Api-Key: {{ auth_secret }}",
        default_extra_headers={"anthropic-version": "2023-06-01"},
        env_var="ANTHROPIC_API_KEY",
    ),
    KnownProvider(
        name="google-gemini",
        label="Google Gemini",
        description="Gemini 2.5 Flash, Pro",
        host_url="https://generativelanguage.googleapis.com/v1beta/openai",
        env_var="GEMINI_API_KEY",
    ),
    KnownProvider(
        name="ollama",
        label="Ollama (local)",
        description="Local models, no API key needed",
        host_url="http://localhost:11434/v1",
        requires_api_key=False,
    ),
)

_KNOWN_PROVIDERS_BY_NAME: dict[str, KnownProvider] = {p.name: p for p in KNOWN_PROVIDERS}


def _provider_type_for_connection(name: str, host_url: str) -> str:
    known = _KNOWN_PROVIDERS_BY_NAME.get(name)
    if known is not None and known.host_url.rstrip("/") == host_url.rstrip("/"):
        return known.name
    return "custom"


_POST_SETUP_OPTIONS: tuple[tuple[str, str], ...] = (
    ("sample", "Create a sample workspace and demo agent"),
    ("explore", "I would like to explore NeMo Helix on my own"),
)


@dataclass(frozen=True)
class ProbeConfig:
    """How to probe a provider's auth-required endpoint for key validation."""

    method: str
    path: str
    body: dict | None = None


# nvidia-build lists the catalog, then POSTs chat to a live candidate. Listing
# alone is unauthenticated; chat is what rejects a bad key.
_PROBE_CONFIGS: dict[str, ProbeConfig] = {
    "nvidia-build": ProbeConfig("GET", "v1/models"),
    "openai": ProbeConfig("GET", "models"),
    "anthropic": ProbeConfig("GET", "v1/models"),
    "google-gemini": ProbeConfig("GET", "models"),
}


class KeyValidationStatus(StrEnum):
    """Outcome categories for an API key validation probe."""

    VALID = "valid"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class KeyValidationResult:
    """Outcome of an API key validation probe."""

    status: KeyValidationStatus
    message: str = ""

    @property
    def passed(self) -> bool:
        """True when setup may continue without treating the key as rejected.

        ``INCONCLUSIVE`` is fail-open (same posture as non-2xx probes for OpenAI
        and other auth-required list endpoints). Only ``REJECTED`` blocks auto setup.
        """
        return self.status != KeyValidationStatus.REJECTED


@dataclass(frozen=True)
class ModelPair:
    """Model entities selected for quality-critical and low-latency agent work."""

    default: str
    fast: str


@dataclass(frozen=True)
class SetupClients:
    """Typed service clients the setup flow talks to, all sharing the CLI's auth and transport."""

    models: ModelsClient
    secrets: SecretsClient
    gateway: InferenceGatewayClient

    @classmethod
    def from_context(cls, cli_context: CLIContext) -> SetupClients:
        return cls(
            models=cli_context.typed_client(ModelsClient),
            secrets=cli_context.typed_client(SecretsClient),
            gateway=cli_context.typed_client(InferenceGatewayClient),
        )


# Env vars probed during --auto mode, in priority order.
_AUTO_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("NEMO_DEFAULT_INFERENCE_KEY", "NEMO_DEFAULT_INFERENCE_BASE_URL"),
    ("NVIDIA_API_KEY", ""),
    ("OPENAI_API_KEY", ""),
    ("ANTHROPIC_API_KEY", ""),
    ("GEMINI_API_KEY", ""),
)

_KEY_VALIDATION_TIMEOUT = 10.0
_KEY_REJECTED_STATUS_CODES = (401, 403)
_KEY_REJECTED_MESSAGE = "API key validation failed. The provider rejected the credentials."
_KEY_UNVERIFIED_SUFFIX = "Credentials were neither accepted nor rejected."
_PROBE_DETAIL_MAX_CHARS = 200
# Key validation uses a smaller candidate budget than post-setup model selection.
_KEY_VALIDATION_CHAT_MAX_ATTEMPTS = 3

_KEY_VALIDATION_ACTIONS: tuple[tuple[str, str], ...] = (
    ("continue", "Continue without verifying the key"),
    ("reenter", "Re-enter API key"),
    ("abort", "Abort setup"),
)


# Catalog listing is not entitlement-scoped; only a successful chat request counts.
_MODEL_PROBE_TIMEOUT = 20.0
_MODEL_PROBE_MAX_ATTEMPTS = 8
_MODEL_PROBE_BUDGET_SECONDS = 90.0
# Gateway 404s until the model's VirtualModel exists and the gateway cache
# (3s) picks it up. The models controller ticks every 5s and a busy step can
# take longer, so a 10s window expires in the same second the route appears.
_MODEL_ROUTE_READY_SECONDS = 30.0
_MODEL_ROUTE_RETRY_INTERVAL = 1.0
_MODEL_ROUTE_MAX_RETRIES = 30

_MODEL_DISCOVERY_ROUND_SECONDS = 30
_MODEL_DISCOVERY_MAX_ROUNDS = 2
_MODEL_DISCOVERY_POLL_INTERVAL = 1
_SERVICE_STARTUP_TIMEOUT_SECONDS = 240
_SERVICE_STARTUP_POLL_INTERVAL = 0.5
_AGENT_DEPLOY_TIMEOUT_SECONDS = 120
_AGENT_DEPLOY_POLL_INTERVAL = 1
_KILL_WAIT_TIMEOUT = 10
_CONTROLLER_HEALTH_RETRY_DELAY = 3.0
_POST_START_REACHABLE_RETRIES = 6
_POST_START_REACHABLE_DELAY = 2.0

_SAMPLE_AGENT_NAME = "email-security-triage"
_SAMPLE_DATASET_FILESET = "esec-eval-data"
_SAMPLE_DATASET_FILENAME = "dataset.jsonl"
_SAMPLE_EVAL_CONFIG_FILENAME = "eval-config.yaml"
_LOCAL_CONTEXT_NAME = "local"


def _pause(seconds: float) -> None:
    time.sleep(seconds)


# Filesystem markers that indicate which coding agents are in use.
_AGENT_MARKERS: tuple[tuple[str, str], ...] = (
    ("AGENTS.md", "codex"),
    (".cursor", "cursor"),
    (".opencode", "opencode"),
    (".claude", "claude"),
)

# ---------------------------------------------------------------------------
# Helpers — platform reachability
# ---------------------------------------------------------------------------


def _bootstrap_config_if_missing(base_url: str, workspace: str) -> None:
    """Write a minimal cluster + context into the config file when one isn't seeded.

    ``nemo setup`` can run before any config is on disk (first-time install)
    *or* with a partial config containing only ``local_services.data_dir``
    written earlier in the same setup invocation by ``_save_data_dir``.
    Later steps — in particular ``_save_model_pair`` — call
    ``Config.write`` with only model-selection params. If there's no
    cluster on disk at that point, ``ensure_context`` will fail with
    ``Cluster '<name>' does not exist and no base_url provided to create
    it`` because it has no ``base_url`` to attach to a new cluster.

    Calling this once after the platform is confirmed reachable seeds the
    cluster + context so that all subsequent writes succeed. Idempotent:
    if a cluster is already present, the call is a no-op. Any existing
    ``local_services`` block (e.g. the persisted data dir) is preserved.
    """
    config_path = Config.get_default_config_path()
    if config_path.exists():
        try:
            existing = Config.load(config_path=config_path).get_config_file()
        except Exception:
            # Unreadable existing config — fall through and rewrite.
            logger.debug("Failed to load existing config; will reseed", exc_info=True)
        else:
            if existing.clusters:
                # Cluster already present — assume bootstrap is done.
                return
    params: ConfigParams = {"base_url": base_url, "workspace": workspace}
    Config.write(params)


_PLATFORM_REACHABILITY_PATHS = ("/status", "/cluster-info")


def _check_platform_reachable(
    base_url: str,
    timeout: float = 5.0,
    *,
    certificate_authority: str | None = None,
) -> bool:
    """Return True if a platform health endpoint responds.

    Local ``nemo services run`` publishes ``/status``. Hosted deployments may
    only expose ``/cluster-info`` on ingress, so try both.
    """
    tls_config = httpx_tls_config_from_env(certificate_authority)
    root = base_url.rstrip("/")
    for path in _PLATFORM_REACHABILITY_PATHS:
        try:
            resp = httpx.get(f"{root}{path}", timeout=timeout, **tls_config)
            if resp.status_code == 200:
                return True
        except Exception:
            continue
    return False


def _check_platform_reachable_with_retries(
    base_url: str,
    retries: int = _POST_START_REACHABLE_RETRIES,
    delay: float = _POST_START_REACHABLE_DELAY,
    *,
    certificate_authority: str | None = None,
) -> bool:
    """Check platform reachability with retries.

    Right after startup the platform may briefly report ready then flip back
    to not-ready while controllers begin heavy work (e.g. model reconciliation).
    A single-shot check can hit this window and falsely report failure.
    """
    for attempt in range(retries):
        if _check_platform_reachable(base_url, certificate_authority=certificate_authority):
            return True
        if attempt < retries - 1:
            _pause(delay)
    return False


def _prompt_remote_base_url(*, default_url: str = "", certificate_authority: str | None = None) -> str:
    """Prompt until the user provides a reachable remote Platform URL."""
    while True:
        base_url = prompt_text(
            "Enter the remote Platform base URL: ",
            default=default_url,
            validator=non_empty_validator("Base URL"),
        ).strip()
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            console.print(f"{CROSS} Enter a valid HTTP or HTTPS URL.")
            continue

        base_url = base_url.rstrip("/")
        if _check_platform_reachable_with_retries(base_url, certificate_authority=certificate_authority):
            return base_url

        console.print(f"{CROSS} Unable to connect to NeMo Helix at {display_url(base_url)}.")


def _resolve_setup_workspace(cli_context: CLIContext, workspace: str | None) -> str:
    """Prefer an explicit ``--workspace``; otherwise keep the active context workspace."""
    if workspace is not None:
        return workspace
    return cli_context.get_sdk_context().workspace or DEFAULT_WORKSPACE


def _configure_remote_connection(cli_context: CLIContext, base_url: str, workspace: str) -> None:
    """Persist a remote Platform URL in the active CLI context."""
    context_name = cli_context.get_sdk_context().context_name
    Config.write(
        {"base_url": base_url, "workspace": workspace},
        context_name=context_name,
    )
    cli_context.overrides["base_url"] = base_url
    cli_context.reset_sdk_context()


def _local_context_name(config_file: ConfigFile) -> str:
    """Reuse a compatible local context or choose a name that cannot overwrite one."""
    for context in config_file.contexts:
        cluster = next(cluster for cluster in config_file.clusters if cluster.name == context.cluster)
        user = next(user for user in config_file.users if user.name == context.user)
        if str(cluster.base_url).rstrip("/") == DEFAULT_BASE_URL.rstrip("/") and isinstance(user, NoAuthUser):
            return context.name

    existing_names = {context.name for context in config_file.contexts}
    if _LOCAL_CONTEXT_NAME not in existing_names:
        return _LOCAL_CONTEXT_NAME
    suffix = 2
    while f"{_LOCAL_CONTEXT_NAME}-{suffix}" in existing_names:
        suffix += 1
    return f"{_LOCAL_CONTEXT_NAME}-{suffix}"


def _configure_local_connection(cli_context: CLIContext, workspace: str) -> None:
    """Activate an isolated no-auth context for local services.

    Reusing an authenticated remote context after changing only its URL leaves
    remote OAuth credentials attached to a local cluster. The SDK then tries to
    refresh those credentials using the local auth discovery response. Keep the
    remote context intact and use a dedicated local context instead.
    """
    context_name = _local_context_name(Config.load().get_config_file())
    params: ConfigParams = {
        "base_url": DEFAULT_BASE_URL,
        "workspace": workspace,
        "access_token": None,
        "refresh_token": None,
        "current_context": context_name,
    }
    Config.write(
        params,
        context_name=context_name,
        set_current_on_create=True,
    )
    cli_context.overrides["base_url"] = DEFAULT_BASE_URL
    cli_context.overrides["current_context"] = context_name
    cli_context.reset_sdk_context()


def _ensure_platform_auth(cli_context: CLIContext) -> None:
    """Authenticate the active context when it lacks usable credentials."""
    from nemo_helix_ext.cli.commands.auth import _login_with_oidc, _runtime_token_source_label

    context = cli_context.get_sdk_context()
    if runtime_token_source := _runtime_token_source_label():
        console.print(f"{CHECK} Using {runtime_token_source}\n")
        return

    authenticated = _login_with_oidc(
        cli_context,
        http_client=cli_context.get_http_client(),
        selected_context=context.context_name,
    )
    if not authenticated:
        Config.write(
            {"access_token": None, "refresh_token": None},
            context_name=context.context_name,
        )
    cli_context.reset_sdk_context()


def _platform_request_headers(cli_context: CLIContext) -> dict[str, str] | None:
    """Return authentication headers for direct Platform HTTP requests."""
    headers = cli_context.get_http_headers()
    return headers if isinstance(headers, dict) else None


def _hosted_platform_without_status(base_url: str, *, timeout: float, tls_config: HttpxTLSConfig) -> bool:
    """Return True when ``/cluster-info`` confirms a hosted platform that omits ``/status``."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/cluster-info", timeout=timeout, **tls_config)
    except Exception:
        return False
    return resp.status_code == 200


def _check_controller_health(
    base_url: str,
    timeout: float = 5.0,
    *,
    certificate_authority: str | None = None,
) -> tuple[bool, str]:
    """Query ``/status`` and assess controller health.

    Returns ``(True, "")`` when controllers are populated and all healthy.
    Returns ``(True, detail)`` when ``/status`` is absent but ``/cluster-info`` confirms a hosted platform.
    Returns ``(False, detail)`` when unhealthy, unreachable, or still unhealthy after retry.

    A controller shows up in ``controllers.status`` as soon as the runner starts
    tracking it — before it has registered its control loop — so an in-progress
    startup can legitimately report unhealthy on the first call, not just an
    empty ``controllers.status``. Either case waits ``_CONTROLLER_HEALTH_RETRY_DELAY``
    seconds and retries once before giving up.
    """
    tls_config = httpx_tls_config_from_env(certificate_authority)
    root = base_url.rstrip("/")
    for attempt in range(2):
        try:
            resp = httpx.get(f"{root}/status", timeout=timeout, **tls_config)
            if resp.status_code == 404:
                if _hosted_platform_without_status(root, timeout=timeout, tls_config=tls_config):
                    return True, "Hosted deployment does not publish /status."
                return False, "Unexpected status 404 from /status endpoint."
            if resp.status_code != 200:
                return False, f"Unexpected status {resp.status_code} from /status endpoint."
            data = resp.json()
        except httpx.RequestError:
            return False, "Could not reach platform status endpoint."
        except ValueError:
            return False, "Invalid JSON from /status endpoint."

        controllers = data.get("controllers") if isinstance(data, dict) else None
        if not isinstance(controllers, dict):
            return False, "Invalid /status payload (missing controllers)."
        status_map = controllers.get("status")
        if not isinstance(status_map, dict):
            status_map = {}

        if status_map:
            unhealthy = [name for name, ok in status_map.items() if not ok]
            if not unhealthy:
                return True, ""
            if attempt == 0:
                _pause(_CONTROLLER_HEALTH_RETRY_DELAY)
                continue
            return False, f"Unhealthy controllers: {', '.join(unhealthy)}"

        if attempt == 0:
            _pause(_CONTROLLER_HEALTH_RETRY_DELAY)

    return False, "No controllers reported status. Controller threads may have crashed before registering."


def _verify_platform_health(base_url: str, *, certificate_authority: str | None = None) -> bool:
    """Final health gate before declaring setup complete.

    Returns True if the platform is healthy (caller prints the success banner).
    Returns False after printing red or yellow diagnostics to the console.
    """
    ok, detail = _check_controller_health(base_url, certificate_authority=certificate_authority)
    if ok:
        if detail:
            if "does not publish /status" in detail.lower():
                # Expected for hosted ingress that only exposes /cluster-info.
                console.print(f"\n{CHECK} {detail}")
            else:
                console.print(f"\n{WARN} [yellow]{detail}[/yellow]")
                console.print("  Setup may have succeeded, but controller health could not be verified.")
        return True

    if "no controllers" in detail.lower():
        console.print(f"\n{WARN} [yellow]Could not confirm controller health ({detail}).[/yellow]")
        console.print("  Setup may have succeeded, but verify with:")
        console.print("    [cyan]nemo services status[/cyan]")
    else:
        console.print(f"\n{CROSS} [red]Platform controllers are unhealthy.[/red]")
        console.print(f"  {detail}")
        console.print()
        console.print("  The models controller may have crashed during startup.")
        console.print("  Check service logs with: [cyan]nemo services logs[/cyan]")
        console.print()
        console.print("  Try: [cyan]nemo services run[/cyan]   (restart services)")

    return False


def _provider_exists(clients: SetupClients, name: str, workspace: str) -> bool:
    """Return True if a provider with *name* already exists."""
    try:
        clients.models.get_provider(name=name, workspace=workspace)
        return True
    except Exception:
        return False


def _secret_exists(clients: SetupClients, name: str, workspace: str) -> bool:
    """Return True if a secret with *name* already exists."""
    try:
        clients.secrets.get_secret(name=name, workspace=workspace)
        return True
    except Exception:
        return False


def _create_secret(clients: SetupClients, name: str, value: str, workspace: str) -> None:
    clients.secrets.create_secret(body=HelixSecretCreateRequest(name=name, value=SecretStr(value)), workspace=workspace)


def _update_secret(clients: SetupClients, name: str, value: str, workspace: str) -> None:
    clients.secrets.update_secret(name=name, body=HelixSecretUpdateRequest(value=SecretStr(value)), workspace=workspace)


def _create_provider(
    clients: SetupClients,
    *,
    name: str,
    host_url: str,
    secret_name: str | None,
    workspace: str,
    auth_header_format: str | None = None,
    default_extra_headers: dict[str, str] | None = None,
) -> None:
    fields: dict[str, Any] = {"name": name, "host_url": host_url}
    if secret_name:
        fields["api_key_secret_name"] = secret_name
    if auth_header_format:
        fields["auth_header_format"] = auth_header_format
    if default_extra_headers:
        fields["default_extra_headers"] = default_extra_headers
    provider_type = _provider_type_for_connection(name, host_url)
    try:
        clients.models.create_provider(body=CreateModelProviderRequest(**fields), workspace=workspace)
    except Exception:
        emit.emit_event(
            OnboardingStepEvent(
                step="provider_connected", task_status=TaskStatusEnum.ERROR, provider_type=provider_type
            )
        )
        raise
    emit.emit_event(
        OnboardingStepEvent(
            step="provider_connected", task_status=TaskStatusEnum.COMPLETED, provider_type=provider_type
        )
    )


def _update_provider(
    clients: SetupClients,
    *,
    name: str,
    host_url: str,
    secret_name: str | None,
    workspace: str,
    auth_header_format: str | None = None,
    default_extra_headers: dict[str, str] | None = None,
) -> None:
    fields: dict[str, Any] = {"host_url": host_url}
    if secret_name:
        fields["api_key_secret_name"] = secret_name
    if auth_header_format:
        fields["auth_header_format"] = auth_header_format
        fields["required_extra_headers"] = None
    if default_extra_headers:
        fields["default_extra_headers"] = default_extra_headers
    provider_type = _provider_type_for_connection(name, host_url)
    try:
        clients.models.upsert_provider(name=name, body=UpsertModelProviderRequest(**fields), workspace=workspace)
    except Exception:
        emit.emit_event(
            OnboardingStepEvent(
                step="provider_connected", task_status=TaskStatusEnum.ERROR, provider_type=provider_type
            )
        )
        raise
    emit.emit_event(
        OnboardingStepEvent(
            step="provider_connected", task_status=TaskStatusEnum.COMPLETED, provider_type=provider_type
        )
    )


_PROVIDER_UNHEALTHY_STATUSES = frozenset({"ERROR", "LOST"})
_NON_COMPLIANT_MARKER = "Non-OpenAI compliant"


def _bucket_model_count(count: int) -> str:
    """Bucket a discovered-model count into a coarse range for telemetry.

    Keeps the emitted value low-cardinality (no exact counts leave the machine).
    """
    if count <= 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    if count <= 50:
        return "21-50"
    if count <= 100:
        return "51-100"
    if count <= 250:
        return "101-250"
    return "251+"


def _wait_for_models(
    clients: SetupClients,
    provider_name: str,
    workspace: str,
    host_url: str = "",
    round_seconds: int = _MODEL_DISCOVERY_ROUND_SECONDS,
    max_rounds: int = _MODEL_DISCOVERY_MAX_ROUNDS,
) -> list[str]:
    """Poll for served models and emit one ``models_discovered`` event.

    Thin telemetry wrapper around :func:`_wait_for_models_impl`: COMPLETED with
    the discovered-count bucket on success, ERROR (re-raised) if polling blows up.
    """
    try:
        models = _wait_for_models_impl(clients, provider_name, workspace, host_url, round_seconds, max_rounds)
    except Exception:
        emit.emit_event(OnboardingStepEvent(step="models_discovered", task_status=TaskStatusEnum.ERROR))
        raise
    emit.emit_event(
        OnboardingStepEvent(
            step="models_discovered",
            task_status=TaskStatusEnum.COMPLETED,
            models_discovered_bucket=_bucket_model_count(len(models)),
        )
    )
    return models


def _wait_for_models_impl(
    clients: SetupClients,
    provider_name: str,
    workspace: str,
    host_url: str = "",
    round_seconds: int = _MODEL_DISCOVERY_ROUND_SECONDS,
    max_rounds: int = _MODEL_DISCOVERY_MAX_ROUNDS,
) -> list[str]:
    """Poll until provider has at least one served model. Returns entity IDs.

    Retries in rounds so the user sees progress rather than a long silence.
    Checks provider status each poll and exits early if the provider is
    flagged as non-compliant or unhealthy, so the user isn't left waiting
    for models that will never arrive.
    """
    start = time.monotonic()
    for attempt in range(max_rounds):
        deadline = time.monotonic() + round_seconds
        with console.status("[bold cyan]Waiting for model discovery...") as status:
            while time.monotonic() < deadline:
                elapsed = int(time.monotonic() - start)
                status.update(f"[bold cyan]Waiting for model discovery... ({elapsed}s)")
                try:
                    provider = clients.models.get_provider(name=provider_name, workspace=workspace).data()
                    served = getattr(provider, "served_models", None) or []
                    if served:
                        model_ids = [m.model_entity_id for m in served if getattr(m, "model_entity_id", None)]
                        if model_ids:
                            return model_ids

                    raw_status = getattr(provider, "status", None)
                    provider_status = raw_status.value if isinstance(raw_status, Enum) else (raw_status or "")
                    provider_msg = getattr(provider, "status_message", None) or ""

                    if _NON_COMPLIANT_MARKER in provider_msg:
                        url_hint = f" ({display_url(host_url)})" if host_url else ""
                        console.print(
                            f"\n  {WARN} Provider '{provider_name}'{url_hint} returned a non-OpenAI "
                            f"compliant response from GET /v1/models."
                        )
                        console.print("  Check that the host URL points to an OpenAI-compatible API endpoint.")
                        console.print(
                            "  The provider is still registered and usable for direct inference, "
                            "but automatic model discovery is disabled."
                        )
                        return []

                    if provider_status in _PROVIDER_UNHEALTHY_STATUSES:
                        url_hint = f" ({display_url(host_url)})" if host_url else ""
                        console.print(f"\n  {WARN} Provider '{provider_name}'{url_hint} is in {provider_status} state.")
                        if provider_msg:
                            console.print(f"  {provider_msg}")
                        console.print("  Check the host URL and API key, then re-run [cyan]nemo setup[/cyan].")
                        return []

                except Exception:
                    logger.debug("Model discovery poll for '%s' failed", provider_name, exc_info=True)
                _pause(_MODEL_DISCOVERY_POLL_INTERVAL)
        if attempt < max_rounds - 1:
            console.print(f"  {WARN} Models not available yet, retrying...")
    return []


def _get_all_model_entity_ids(
    clients: SetupClients,
    workspace: str,
    *,
    provider_name: str | None = None,
) -> list[str]:
    """Return model entity IDs, optionally scoped to one provider."""
    entity_ids: list[str] = []
    try:
        for provider in clients.models.list_providers(workspace=workspace).items():
            if provider_name is not None and getattr(provider, "name", None) != provider_name:
                continue
            for model in getattr(provider, "served_models", None) or []:
                if hasattr(model, "model_entity_id") and model.model_entity_id:
                    entity_ids.append(model.model_entity_id)
    except Exception:
        logger.debug("Failed to list model entity IDs", exc_info=True)
    return sorted(set(entity_ids))


_NON_CHAT_MODEL_MARKERS = (
    "embed",
    "embedding",
    "vision",
    "guard",
    "rerank",
    "fuyu",
    "reward",
    "translate",
)


def _is_usable_chat_model_entity(entity_id: str) -> bool:
    """Return whether an entity ID looks like a chat LLM rather than a specialist model."""
    haystack = entity_id.lower().replace("_", "-")
    return not any(marker in haystack for marker in _NON_CHAT_MODEL_MARKERS)


def _pick_default_chat_entity(entity_ids: list[str]) -> str | None:
    """Pick the first discovered chat model, skipping embedding/vision/guard/rerank names.

    ``entity_ids`` is already scoped to a provider's ``served_models``.
    """
    for entity_id in entity_ids:
        if _is_usable_chat_model_entity(entity_id):
            return entity_id
    return None


_MODEL_SIZE_PATTERN = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])")


def _model_parameter_size(entity_id: str) -> float | None:
    """Return the largest ``Nb`` parameter count parsed from an entity name."""
    name = _display_model_name(entity_id).lower().replace("_", "-")
    sizes = [float(match) for match in _MODEL_SIZE_PATTERN.findall(name)]
    return max(sizes) if sizes else None


def _is_preferred_vendor(entity_id: str) -> bool:
    """Return whether the entity name identifies an NVIDIA-published model."""
    name = _display_model_name(entity_id).lower().replace("_", "-")
    return name.startswith(("nvidia-", "nvidia/", "nv-", "nv/")) or "nemotron" in name


def _order_candidates_by_size(entity_ids: list[str], *, largest_first: bool) -> list[str]:
    """Order NVIDIA models first, then by parsed size; unnamed sizes sort last."""

    def sort_key(entity_id: str) -> tuple[bool, bool, float, str]:
        size = _model_parameter_size(entity_id)
        ranked = 0.0 if size is None else (-size if largest_first else size)
        return (not _is_preferred_vendor(entity_id), size is None, ranked, entity_id)

    return sorted(entity_ids, key=sort_key)


def _probe_model_entity(gateway: InferenceGatewayClient, workspace: str, entity_id: str) -> bool:
    """Return True when a short chat request against *entity_id* succeeds.

    Retries route 404s; timeouts skip; connection errors raise.
    """
    route_ready_deadline = time.monotonic() + _MODEL_ROUTE_READY_SECONDS
    retries = 0
    while True:
        try:
            gateway.openai_post(
                trailing_uri="v1/chat/completions",
                workspace=workspace,
                body=JsonBody(
                    {
                        "model": entity_id,
                        "messages": [{"role": "user", "content": "Respond with 'OK'"}],
                        "max_tokens": 16,
                    }
                ),
            )
        except NemoTransportError as exc:
            if not isinstance(exc.error, httpx.TimeoutException):
                raise
            logger.debug("Model probe for '%s' timed out", entity_id, exc_info=True)
            console.print(f"  {WARN} Skipping {_display_model_name(entity_id)} (timed out)")
            return False
        except NemoHTTPError as exc:
            detail = str(exc.body or exc.http_response.text or exc)
            entitlement_miss = "not found for account" in detail.lower()
            if (
                exc.status_code == 404
                and not entitlement_miss
                and retries < _MODEL_ROUTE_MAX_RETRIES
                and time.monotonic() < route_ready_deadline
            ):
                retries += 1
                _pause(_MODEL_ROUTE_RETRY_INTERVAL)
                continue
            console.print(f"  {WARN} Skipping {_display_model_name(entity_id)} (HTTP {exc.status_code})")
            return False
        except Exception as exc:
            logger.debug("Model probe for '%s' failed", entity_id, exc_info=True)
            console.print(f"  {WARN} Skipping {_display_model_name(entity_id)} ({exc})")
            return False
        return True


def _select_usable_model_pair(clients: SetupClients, workspace: str, entity_ids: list[str]) -> ModelPair | None:
    """Choose the largest and smallest models that answer a chat request.

    Returns ``None`` when nothing answers, including when the gateway is
    unreachable, so setup does not persist an unusable default.
    """
    candidates = [entity_id for entity_id in entity_ids if _is_usable_chat_model_entity(entity_id)]
    if not candidates:
        return None

    usable: dict[str, bool] = {}
    probe_client = clients.gateway.with_options(retry=RetryPolicy(max_retries=0), timeout=_MODEL_PROBE_TIMEOUT)
    started = time.monotonic()
    deadline = started + _MODEL_PROBE_BUDGET_SECONDS

    def first_usable(ordered: list[str]) -> str | None:
        attempts = 0
        for entity_id in ordered:
            if entity_id not in usable:
                if attempts >= _MODEL_PROBE_MAX_ATTEMPTS or time.monotonic() >= deadline:
                    return None
                attempts += 1
                usable[entity_id] = _probe_model_entity(probe_client, workspace, entity_id)
            if usable[entity_id]:
                return entity_id
        return None

    console.print("  Verifying models with a short inference request...")
    try:
        default_model = first_usable(_order_candidates_by_size(candidates, largest_first=True))
        if default_model is None:
            return None
        fast_model = first_usable(_order_candidates_by_size(candidates, largest_first=False)) or default_model
    except NemoTransportError as exc:
        console.print(f"  {WARN} Could not verify models ({exc}).")
        return None
    return ModelPair(default=default_model, fast=fast_model)


def _get_all_model_choices(
    clients: SetupClients,
    workspace: str,
    *,
    provider_name: str | None = None,
) -> list[tuple[str, str]]:
    """Return picker choices, optionally scoped to one provider."""
    choices: list[tuple[str, str]] = []
    try:
        for provider in clients.models.list_providers(workspace=workspace).items():
            current_provider_name = getattr(provider, "name", "unknown-provider")
            if provider_name is not None and current_provider_name != provider_name:
                continue
            for model in getattr(provider, "served_models", None) or []:
                model_entity_id = getattr(model, "model_entity_id", None)
                if model_entity_id:
                    label = f"{_display_model_name(model_entity_id)} ({current_provider_name})"
                    choices.append((model_entity_id, label))
    except Exception:
        logger.debug("Failed to list model choices", exc_info=True)
    return sorted(set(choices), key=lambda item: item[1])


def _display_model_name(model_entity_id: str) -> str:
    """Strip workspace prefix from a model entity ID for display."""
    return model_entity_id.split("/", 1)[-1] if "/" in model_entity_id else model_entity_id


def _resolve_provider_for_url(base_url: str) -> KnownProvider | None:
    """Find a known provider whose host_url matches *base_url*."""
    normalized = base_url.rstrip("/")
    for p in KNOWN_PROVIDERS:
        if p.host_url.rstrip("/") == normalized:
            return p
    return None


# ---------------------------------------------------------------------------
# Service startup
# ---------------------------------------------------------------------------


def _load_persisted_data_dir() -> str | None:
    """Return the local data directory previously chosen via ``nemo setup``."""
    config_path = Config.get_default_config_path()
    if not config_path.exists():
        return None
    try:
        config = Config.load(config_path=config_path)
    except Exception:
        logger.debug("Failed to load config for local_services lookup", exc_info=True)
        return None
    local = config.get_config_file().local_services
    return local.data_dir if local else None


def _save_data_dir(data_dir: str) -> None:
    """Persist the chosen local data directory to the user's config file.

    Reads any existing config so other fields (contexts, clusters, etc.) are
    preserved, then overwrites just ``local_services.data_dir``.
    """
    config_path = Config.get_default_config_path()
    if config_path.exists():
        config = Config.load(config_path=config_path)
    else:
        config = Config.create(config_path, ConfigFile())
    config_file = config.get_config_file()
    config_file.local_services = LocalServicesConfig(data_dir=data_dir)
    config.save()


def _prompt_data_dir() -> str:
    """Prompt the user for a local data directory and persist the choice.

    Pre-fills the prompt with the previously-persisted directory if any,
    otherwise the XDG default (``~/.local/share/nemo``).  The chosen
    directory is where local services persist SQLite DB, encryption key,
    and uploaded files.
    """
    persisted = _load_persisted_data_dir()
    default_dir = persisted or str(nhx_user_data_dir())

    chosen = prompt_text(
        message="Local data directory:",
        default=default_dir,
        validator=non_empty_validator("Data directory"),
    ).strip()

    _save_data_dir(chosen)
    return chosen


def _resolve_services_port(base_url: str) -> int:
    """Extract the port from *base_url*, defaulting to 8080."""
    parsed = urlparse(base_url)
    return parsed.port or 8080


def _is_local_base_url(base_url: str) -> bool:
    """Return whether *base_url* points at the local machine."""
    parsed = urlparse(base_url)
    return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _platform_host_label(base_url: str) -> str:
    """Return a human-friendly host label for connection prompts."""
    parsed = urlparse(base_url)
    return parsed.hostname or base_url.rstrip("/")


class _RemoteConnectionChoice(StrEnum):
    """Options offered when a configured remote Platform is already reachable."""

    CONTINUE = "continue"
    START_LOCAL = "local"
    CHANGE_REMOTE = "change"


def _prompt_reachable_remote_connection(base_url: str) -> Literal["ready", "connect_remote", "start_local"]:
    """Ask how to proceed when a configured remote Platform is already reachable."""
    hostname = _platform_host_label(base_url)
    action = prompt_choice(
        message=f"Platform reachable at {hostname} ({display_url(base_url)}). What would you like to do?",
        options=[
            (_RemoteConnectionChoice.CONTINUE, "Continue with this remote Platform"),
            (_RemoteConnectionChoice.START_LOCAL, "Start local services instead"),
            (_RemoteConnectionChoice.CHANGE_REMOTE, "Connect to a different remote URL"),
        ],
        default=_RemoteConnectionChoice.CONTINUE,
    )
    if action == _RemoteConnectionChoice.CONTINUE:
        console.print(f"{CHECK} Platform already running at {display_url(base_url)}\n")
        return "ready"
    if action == _RemoteConnectionChoice.CHANGE_REMOTE:
        return "connect_remote"
    return "start_local"


def _start_services_background(base_url: str, data_dir: str | None = None) -> subprocess.Popen:
    """Launch ``nemo services run`` as a background process.

    Delegates to the shared process lifecycle module which uses flock-based
    instance tracking.  If *data_dir* is provided, it's forwarded so the
    subprocess inherits ``NHX_DATA_DIR`` (unless the parent shell already
    exported it).
    """
    port = _resolve_services_port(base_url)
    return start_background(
        HelixAppConfig(scope=compute_scope(port=port), port=port),
        data_dir=data_dir,
    )


def _last_startup_service(log_path: Path | None) -> str:
    """Read the most recently logged ``[STARTUP] service:<name>`` from the service log."""
    if log_path is None or not log_path.exists():
        return ""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return ""
    last = ""
    tag = "[STARTUP] service:"
    for line in text.splitlines():
        if tag in line:
            last = line.split(tag, 1)[1].split(":")[0]
    return last


def _wait_for_platform(
    base_url: str,
    timeout: int = _SERVICE_STARTUP_TIMEOUT_SECONDS,
    poll_interval: float = _SERVICE_STARTUP_POLL_INTERVAL,
    log_path: Path | None = None,
    proc: subprocess.Popen | None = None,
    *,
    certificate_authority: str | None = None,
) -> bool:
    """Poll until the platform health endpoint responds. Returns True on success.

    When *log_path* is provided, the spinner shows the last service that
    finished loading so users see that progress is being made during a
    slow cold start.

    When *proc* is provided, return False immediately if the service process
    exits before the platform becomes ready (avoid waiting the full timeout).
    """
    start = time.monotonic()
    deadline = start + timeout
    with console.status("[bold cyan]Waiting for platform...") as status:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                return False
            elapsed = int(time.monotonic() - start)
            svc = _last_startup_service(log_path)
            hint = f" — loaded {svc}" if svc else ""
            status.update(f"[bold cyan]Waiting for platform... ({elapsed}s){hint}")
            if _check_platform_reachable(base_url, timeout=1.0, certificate_authority=certificate_authority):
                return True
            _pause(poll_interval)
    return False


_DOCKER_FAILURE_LOG_MARKERS = (
    "Docker daemon is unavailable",
    "Docker is unavailable",
    "docker.from_env",
    "Error while fetching server API version",
)

_PORT_CONFLICT_LOG_MARKERS = (
    "EADDRINUSE",
    "address already in use",
    "address is already in use",
    "Errno 98",
    "Only one usage of each socket address",
)


def _services_log_suggests_docker_failure(log_path: Path | None) -> bool:
    """Return True when services.log contains known Docker skip / daemon errors."""
    if log_path is None or not log_path.is_file():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in text for marker in _DOCKER_FAILURE_LOG_MARKERS)


def _services_log_suggests_port_conflict(log_path: Path | None) -> bool:
    """Return True when services.log contains known TCP bind failure markers."""
    if log_path is None or not log_path.is_file():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    lower_text = text.lower()
    return any(marker.lower() in lower_text for marker in _PORT_CONFLICT_LOG_MARKERS)


def _detect_startup_port_conflict(base_url: str, log_path: Path | None) -> PortConflict | None:
    """Detect a startup-time port conflict after the services process exits.

    The preflight normally catches busy ports before spawning services. This
    fallback handles races where the port becomes busy between preflight and
    uvicorn binding, or where the service process writes the bind failure to
    ``services.log`` before exiting.
    """
    port = _resolve_services_port(base_url)
    scope = compute_scope(port=port)
    conflict = check_port_available_for_start(DEFAULT_LOCAL_SERVICES_BIND_HOST, port, scope)
    if conflict is not None:
        return conflict
    if _services_log_suggests_port_conflict(log_path):
        return PortConflict(kind="foreign", port=port)
    return None


def _should_hint_docker_unavailable(*, exit_code: int | None, log_path: Path | None) -> bool:
    """Decide whether to print a Docker-missing hint after a failed startup wait.

    Prefer log evidence. Otherwise only hint when the process exited early and
    a Docker ping confirms the daemon is unavailable — not on a pure readiness
    timeout while the process is still alive.
    """
    if _services_log_suggests_docker_failure(log_path):
        return True
    if exit_code is not None and not probe_docker(use_cache=False).available:
        return True
    return False


def _kill_existing_services(base_url: str) -> None:
    """Find and kill any running ``nemo services run`` processes.

    Delegates to the shared process lifecycle module.
    """
    stop_instance(compute_scope(port=_resolve_services_port(base_url)), timeout=2.0, force=True)


def _ensure_port_available_for_start(base_url: str) -> None:
    """Fail fast when the services port cannot be bound."""
    port = _resolve_services_port(base_url)
    conflict = check_port_available_for_start(DEFAULT_LOCAL_SERVICES_BIND_HOST, port, compute_scope(port=port))
    if conflict is None:
        return
    lines = format_port_conflict(conflict)
    console.print(f"{CROSS} {lines[0]}")
    for line in lines[1:]:
        console.print(f"  {line}")
    raise typer.Exit(1)


def _maybe_start_services(
    base_url: str,
    auto: bool,
    start_services: bool | None,
    timeout: int = _SERVICE_STARTUP_TIMEOUT_SECONDS,
    *,
    certificate_authority: str | None = None,
) -> Literal["ready", "connect_remote", "start_local"]:
    """Start services if requested, restarting if already running.

    In interactive mode (auto=False), prompts the user if start_services is None.
    In auto mode, only starts if start_services is explicitly True.

    When start_services is True and the platform is already running, the
    existing processes are stopped and restarted so the full service set
    (including any newly installed plugins) is picked up. Data lives in
    SQLite so nothing is lost across restarts.
    """
    if start_services is True and not _is_local_base_url(base_url):
        raise typer.BadParameter(
            "--start-services requires a local Platform URL",
            param_hint="--start-services",
        )

    already_running = _check_platform_reachable(base_url, certificate_authority=certificate_authority)

    if already_running and start_services is not True:
        if _is_local_base_url(base_url) or auto:
            console.print(f"{CHECK} Platform already running at {display_url(base_url)}\n")
            return "ready"
        return _prompt_reachable_remote_connection(base_url)

    should_start = start_services
    if should_start is None:
        if auto:
            console.print(f"{CROSS} Cannot reach platform at {display_url(base_url)}")
            console.print("  Start the platform first, or pass --start-services:")
            console.print("    [cyan]nemo setup --auto --start-services[/cyan]")
            console.print("    [cyan]nemo services run[/cyan]")
            raise typer.Exit(1)
        if not _is_local_base_url(base_url):
            return "connect_remote"
        action = prompt_choice(
            message=f"Platform not reachable at {display_url(base_url)}. Start local services?",
            options=[
                ("yes", "Yes, start services now"),
                ("remote", "No, I want to connect to a remote Platform instance"),
                ("manual", "No, I'll start them myself"),
            ],
            default="yes",
        )
        if action == "remote":
            return "connect_remote"
        should_start = action == "yes"

    if not should_start:
        console.print(f"{CROSS} Cannot reach platform at {display_url(base_url)}")
        console.print("  Start the platform first:")
        console.print("    [cyan]nemo services run[/cyan]   (local development)")
        raise typer.Exit(1)

    # Pick (and persist) the local data directory before launching services
    # so the chosen path takes effect on this run.  Interactive mode prompts;
    # ``--auto`` reuses whatever was previously persisted (or service default).
    if auto:
        data_dir = _load_persisted_data_dir()
    else:
        data_dir = _prompt_data_dir()

    if importlib.util.find_spec("pyleak") is None:
        console.print(f"{CROSS} Local services require extra dependencies that aren't installed.")
        console.print("  Install them with:")
        console.print(f"    [cyan]{escape(services_extra_install_command())}[/cyan]")
        raise typer.Exit(1)

    # Fail before stop/spawn when default local needs Docker (NVBug 6537617).
    require_docker_for_default_local(console=console)

    if already_running:
        console.print("  Restarting Helix services...")
        _kill_existing_services(base_url)
        deadline = time.time() + _KILL_WAIT_TIMEOUT
        while time.time() < deadline and _check_platform_reachable(
            base_url,
            timeout=1.0,
            certificate_authority=certificate_authority,
        ):
            _pause(1)
    else:
        console.print("  Starting Helix services...")
    _ensure_port_available_for_start(base_url)
    proc = _start_services_background(base_url, data_dir=data_dir)

    log = log_path_for(compute_scope(port=_resolve_services_port(base_url)))

    if not _wait_for_platform(
        base_url,
        timeout=timeout,
        log_path=log,
        proc=proc,
        certificate_authority=certificate_authority,
    ):
        exit_code = proc.poll()
        startup_conflict = _detect_startup_port_conflict(base_url, log) if exit_code is not None else None
        if startup_conflict is not None:
            lines = format_port_conflict(startup_conflict)
            console.print(f"{CROSS} {lines[0]}")
            for line in lines[1:]:
                console.print(f"  {line}")
        elif exit_code is not None:
            console.print(f"{CROSS} Service process exited early (exit code {exit_code})")
        else:
            proc.terminate()
            console.print(f"{CROSS} Platform did not become ready within {timeout}s")
        console.print(f"  Check {log} for details.")
        if _should_hint_docker_unavailable(exit_code=exit_code, log_path=log):
            console.print(f"  {DOCKER_PREFLIGHT_MESSAGE}")
        raise typer.Exit(1)

    console.print(f"{CHECK} Platform running at {display_url(base_url)} (pid {proc.pid})\n")
    return "ready"


# ---------------------------------------------------------------------------
# Skills installation
# ---------------------------------------------------------------------------


def _detect_coding_agents() -> list[tuple[str, str]]:
    """Detect coding agents from filesystem markers in the project root.

    Returns list of (marker, agent_name) for each detected agent.
    """
    project_root = _find_project_root()
    detected = []
    for marker, agent_name in _AGENT_MARKERS:
        if (project_root / marker).exists():
            detected.append((marker, agent_name))
    return detected


def _find_project_root() -> Path:
    """Find the project root by looking for a .git directory, falling back to cwd."""
    cwd = Path.cwd()
    current = cwd
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return cwd


def _load_skills_with_warnings() -> tuple[dict[str, Skill], list[str]]:
    """Load skills while capturing any plugin-discovery warnings.

    The registry emits ``logger.warning`` records when a plugin's skills directory
    is missing, malformed, or invalid. We capture those records so they can be
    surfaced in the preview before any install happens, rather than scrolling
    past mid-install.

    Defensive against (a) callers that raised the registry logger's level above
    WARNING (e.g. a future ``--quiet`` flag) and (b) the ``@lru_cache`` on the
    underlying loader, which would otherwise replay a cached dict with no
    warnings on repeat calls.
    """
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _Capture(level=logging.WARNING)
    original_level = skills_registry.logger.level
    skills_registry.logger.addHandler(handler)
    if original_level > logging.WARNING or original_level == logging.NOTSET:
        skills_registry.logger.setLevel(logging.WARNING)
    skills_registry.clear_cache()
    try:
        skills = load_skills()
    finally:
        skills_registry.logger.removeHandler(handler)
        skills_registry.logger.setLevel(original_level)
    return skills, captured


def _print_plugin_warnings(plugin_warnings: list[str]) -> None:
    """Surface plugin-discovery warnings before the interactive skills prompt."""
    if not plugin_warnings:
        return
    console.print("  [yellow]Plugin warnings:[/yellow]")
    for msg in plugin_warnings:
        console.print(f"    {WARN} {msg}")
    console.print()


_BUILTIN_SOURCE_NAME = "nemo-helix"
_CUSTOM_AGENT_NAME = "other"


def _get_skills_installer(agent_name: str) -> BaseAgentInstaller:
    if agent_name == _CUSTOM_AGENT_NAME:
        return CustomPathInstaller()
    return get_installer(agent_name)


def _skill_sources_of(skills: dict[str, Skill]) -> dict[str, list[Skill]]:
    """Group skills by their source (built-in vs each plugin).

    The built-in source is keyed by ``nemo-helix``; plugin skills are keyed
    by ``Skill.source_plugin``. Returned dict preserves insertion order so the
    built-in group appears first, then plugins in discovery order.
    """
    sources: dict[str, list[Skill]] = {}
    for skill in skills.values():
        key = skill.source_plugin or _BUILTIN_SOURCE_NAME
        sources.setdefault(key, []).append(skill)
    return sources


def _filter_agents_by_scope(agents: list[str], scope: Scope) -> tuple[list[str], list[tuple[str, str]]]:
    """Split agents into (installable, skipped) based on whether each supports the chosen scope.

    Returns:
        (kept, skipped) where ``skipped`` is a list of (agent_name, reason) tuples.
    """
    kept: list[str] = []
    skipped: list[tuple[str, str]] = []
    for agent in agents:
        installer = _get_skills_installer(agent)
        if scope in installer.supported_scopes:
            kept.append(agent)
        else:
            supported = ", ".join(s.value for s in installer.supported_scopes) or "none"
            skipped.append((agent, f"does not support '{scope.value}' scope (supports: {supported})"))
    return kept, skipped


def _print_final_skills_summary(
    agents: list[str], scope: Scope, skill_names: list[str], custom_path: Path | None = None
) -> None:
    """Print the planned action before final confirmation."""
    if not skill_names or not agents:
        return
    project_root = _find_project_root()
    agent_list = ", ".join(agents)
    console.print(
        f"  Installing [bold]{len(skill_names)}[/bold] skill(s) for "
        f"[bold]{agent_list}[/bold] at [bold]{scope.value}[/bold] scope:"
    )
    for agent in agents:
        installer = _get_skills_installer(agent)
        # Show the parent directory of one representative skill so the user
        # sees the destination root, not a single SKILL.md path.
        install_root = custom_path if agent == _CUSTOM_AGENT_NAME and custom_path is not None else project_root
        example = installer.get_install_path(scope, install_root, skill_names[0])
        console.print(f"    {agent} → {example.parent.parent}/")
    console.print()


def _run_skill_install(
    *,
    agents: list[str],
    scope: Scope,
    skill_names: list[str],
    all_skills: dict[str, Skill],
    project_root: Path,
    custom_path: Path | None = None,
) -> None:
    """Run the actual installer for each agent with the chosen skill subset.

    Skill names are expected to come from a validated source-selection path
    (interactive multiselect or ``--skills-from`` flag), so any name not in
    ``all_skills`` would be an internal bug. If every agent's install fails,
    raises ``typer.Exit(1)`` so callers in ``--auto`` see a non-zero exit.
    """
    chosen = {name: all_skills[name] for name in skill_names if name in all_skills}
    if not chosen:
        console.print(f"  {WARN} No skills selected to install.")
        return

    skills_target = ",".join(agents)
    try:
        successes = 0
        failures = 0
        for agent in agents:
            try:
                installer = _get_skills_installer(agent)
                install_root = custom_path if agent == _CUSTOM_AGENT_NAME and custom_path is not None else project_root
                installer.install(scope, install_root, chosen)
                console.print(f"  {CHECK} Installed {len(chosen)} skill(s) for {agent}")
                successes += 1
            except Exception as exc:
                console.print(f"  {WARN} Failed to install skills for {agent}: {exc}")
                failures += 1

        if failures and not successes:
            raise typer.Exit(1)
    except Exception:
        emit.emit_event(
            OnboardingStepEvent(step="skills_installed", task_status=TaskStatusEnum.ERROR, skills_target=skills_target)
        )
        raise
    emit.emit_event(
        OnboardingStepEvent(step="skills_installed", task_status=TaskStatusEnum.COMPLETED, skills_target=skills_target)
    )


def _parse_csv_flag(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI flag into a list, dropping empty entries."""
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    cleaned = [p for p in parts if p]
    return cleaned or None


def _maybe_install_skills(
    auto: bool,
    install_skills: bool | None,
    *,
    skills_agents: list[str] | None = None,
    skills_scope: Scope | None = None,
    skills_from: list[str] | None = None,
    skills_path: Path | None = None,
) -> None:
    """Install coding agent skills if requested.

    ``--install-skills`` is the master opt-in. ``--skills-agents``,
    ``--skills-scope``, and ``--skills-from`` are filters that narrow what
    gets installed when the master is set; on their own they do nothing in
    non-interactive mode. This mirrors ``_maybe_start_services``, which
    requires its boolean master flag to be explicitly True under ``--auto``.

    ``--skills-from`` selects by *source* (the built-in ``nemo-helix`` set
    or a plugin name) rather than by individual skill name. Picking one source
    installs every skill that source provides.

    Interactive mode walks the user through a source multi-select (each source
    expanded to show its skills as read-only sub-labels, with ``s`` to skip the
    whole step), an agent multi-select, a scope choice, and a final
    confirmation; filter flags pre-populate the defaults.
    """
    if install_skills is False:
        return

    # Validate --skills-agents up-front: a typo like `--skills-agents copex` should
    # fail loudly before any platform work, regardless of detection state.
    if skills_path is not None:
        skills_path = skills_path.expanduser().resolve()
        if skills_agents is not None:
            raise typer.BadParameter("--skills-path and --skills-agents cannot be combined", param_hint="--skills-path")
        skills_agents = [_CUSTOM_AGENT_NAME]

    if skills_agents:
        for agent in skills_agents:
            _get_skills_installer(agent)

    detected = _detect_coding_agents()
    # --skills-agents overrides detection: an explicit instruction to install for
    # an agent wins over "we didn't find a marker file for it." Detection is still
    # load-bearing as the default when the flag is absent.
    all_skills, plugin_warnings = _load_skills_with_warnings()
    if not all_skills:
        console.print(f"  {WARN} No NeMo skills available to install.")
        return

    sources = _skill_sources_of(all_skills)
    source_names = list(sources.keys())

    # Validate --skills-from up-front, same shape as --skills-agents.
    if skills_from:
        unknown = [s for s in skills_from if s not in sources]
        if unknown:
            known = ", ".join(source_names)
            raise typer.BadParameter(
                f"Unknown skill source(s): {', '.join(unknown)}. Known sources: {known}",
                param_hint="--skills-from",
            )

    detected_names = [name for _, name in detected]
    # Detection determines defaults only. Every built-in target remains available
    # so setup works outside an existing coding-agent project.
    menu_agent_names = [*list_agent_names(), _CUSTOM_AGENT_NAME]
    project_root = _find_project_root()
    non_interactive = auto or not is_interactive()

    def _skills_for_sources(chosen_sources: list[str]) -> list[str]:
        return [skill.name for source in chosen_sources for skill in sources[source]]

    if non_interactive:
        # Non-interactive path requires the master switch to be explicitly True.
        # Filter flags alone don't opt in (matches --start-services).
        if install_skills is not True:
            return
        chosen_agents = skills_agents or detected_names
        if not chosen_agents:
            console.print(f"  {WARN} No coding agents detected. Use --skills-agents or --skills-path <directory>.")
            return
        if _CUSTOM_AGENT_NAME in chosen_agents and skills_path is None:
            raise typer.BadParameter(
                "The 'other' agent requires --skills-path in non-interactive mode",
                param_hint="--skills-path",
            )
        chosen_scope = skills_scope or Scope.PROJECT
        chosen_sources = skills_from or source_names
        chosen_skills = _skills_for_sources(chosen_sources)
        chosen_agents, skipped = _filter_agents_by_scope(chosen_agents, chosen_scope)
        for agent, reason in skipped:
            console.print(f"  {WARN} Skipping {agent}: {reason}")
        if not chosen_agents:
            console.print(f"  {WARN} No installable agents for scope '{chosen_scope.value}'.")
            return
        _run_skill_install(
            agents=chosen_agents,
            scope=chosen_scope,
            skill_names=chosen_skills,
            all_skills=all_skills,
            project_root=project_root,
            custom_path=skills_path,
        )
        return

    # Interactive path: sources → agents → scope → confirm.
    _print_plugin_warnings(plugin_warnings)

    source_defaults = skills_from if skills_from else source_names
    source_options = [(name, f"{name} (built-in)" if name == _BUILTIN_SOURCE_NAME else name) for name in source_names]
    source_sub_labels = {name: [skill.name for skill in sources[name]] for name in source_names}
    chosen_sources = prompt_multiselect(
        message="Install skills:",
        options=source_options,
        defaults=source_defaults,
        sub_labels=source_sub_labels,
        min_choices=1,
        allow_skip=True,
        indent=2,
    )
    if chosen_sources is None:
        console.print(f"  {WARN} Skipping skill installation.")
        return
    chosen_skills = _skills_for_sources(chosen_sources)

    agent_defaults = skills_agents if skills_agents else detected_names
    chosen_agents = prompt_multiselect(
        message="Install skills for which agents?",
        options=[(name, _get_skills_installer(name).display_name) for name in menu_agent_names],
        defaults=agent_defaults,
        min_choices=1,
        indent=2,
    )

    if _CUSTOM_AGENT_NAME in chosen_agents and skills_path is None:
        skills_path = (
            Path(
                prompt_text(
                    "Skills directory: ",
                    validator=non_empty_validator("skills directory"),
                    hint="For example: ~/.my-agent/skills",
                    indent=2,
                )
            )
            .expanduser()
            .resolve()
        )

    scope_default = (skills_scope or Scope.PROJECT).value
    chosen_scope = Scope(
        prompt_choice(
            message="Install scope:",
            options=[
                (Scope.PROJECT.value, "Local (this repo: .agents/, .cursor/, .claude/, ...)"),
                (Scope.USER.value, "Global (user home: ~/.agents/, ~/.claude/, ...)"),
            ],
            default=scope_default,
            indent=2,
        )
    )

    chosen_agents, skipped = _filter_agents_by_scope(chosen_agents, chosen_scope)
    for agent, reason in skipped:
        console.print(f"  {WARN} Skipping {agent}: {reason}")
    if not chosen_agents:
        console.print(f"  {WARN} No installable agents for scope '{chosen_scope.value}'.")
        return

    _print_final_skills_summary(chosen_agents, chosen_scope, chosen_skills, custom_path=skills_path)
    if not prompt_confirm("Proceed?", default=True, indent=2):
        return

    _run_skill_install(
        agents=chosen_agents,
        scope=chosen_scope,
        skill_names=chosen_skills,
        all_skills=all_skills,
        project_root=project_root,
        custom_path=skills_path,
    )


# ---------------------------------------------------------------------------
# Sample agent setup
# ---------------------------------------------------------------------------


def _print_sample_setup_complete(base_url: str, workspace: str, *, complete: bool) -> None:
    """Print the sample workspace completion card."""
    studio_url = f"{display_url(base_url)}/studio/workspaces/{workspace}/dashboard"
    remove_command = f"nemo workspaces delete {workspace}"
    if complete:
        status = f"{CHECK} [green bold]Sample workspace ready[/green bold]"
        message = "Explore the sample agent, dataset, and evaluation configuration in Studio."
        border_style = "green"
    else:
        status = f"{WARN} [yellow bold]Sample workspace setup incomplete[/yellow bold]"
        message = "Review the warnings above, then run [cyan]nemo setup[/cyan] again to retry."
        border_style = "yellow"
    lines = [
        status,
        "",
        message,
        "",
        f"[bold]Studio:[/bold] [link={studio_url}]{studio_url}[/link]",
        f"[bold]Remove:[/bold] [cyan]{remove_command}[/cyan]",
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Sample agent[/bold]",
            title_align="left",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 1),
        )
    )


def _wait_for_sample_deployment(agents_client: AgentsClient, sample: SampleAgentResponse, *, submitted: bool) -> bool:
    """Wait for the deployment returned by the sample-agent API to reach running."""
    status = sample.deployment_status
    deadline = time.monotonic() + _AGENT_DEPLOY_TIMEOUT_SECONDS
    activity = "Deploying" if submitted else "Waiting for"
    with console.status(f"[bold cyan]{activity} agent '{sample.agent}'..."):
        while status not in {"running", "failed"} and time.monotonic() < deadline:
            try:
                status = agents_client.get_deployment(workspace=sample.workspace, name=sample.deployment).data().status
            except Exception:
                logger.debug("Sample agent deployment status poll failed", exc_info=True)
            if status not in {"running", "failed"}:
                _pause(_AGENT_DEPLOY_POLL_INTERVAL)

    if status == "running":
        if submitted:
            console.print(f"  {CHECK} Deployed agent '{sample.agent}'")
        else:
            console.print(f"  {CHECK} Agent '{sample.agent}' is running")
        return True
    if status == "failed":
        console.print(f"  {CROSS} Agent deployment failed")
    else:
        console.print(f"  {WARN} Agent deployment did not reach running state within {_AGENT_DEPLOY_TIMEOUT_SECONDS}s")
    return False


def _print_sample_progress(event: SampleAgentStreamEvent) -> None:
    """Render a provisioning frame without claiming an existing resource was created."""
    if event.component == "workspace":
        if event.status in {"created", "existing"}:
            action = "Created" if event.status == "created" else "Found existing"
            console.print(f"  {CHECK} {action} sample workspace '{event.workspace}'")
    elif event.component == "agent":
        if event.status in {"created", "existing"}:
            action = "Created" if event.status == "created" else "Found existing"
            console.print(f"  {CHECK} {action} agent '{_SAMPLE_AGENT_NAME}'")
    elif event.component == "deployment":
        if event.status in {"submitted", "existing"}:
            action = "Submitted" if event.status == "submitted" else "Found existing"
            console.print(f"  {CHECK} {action} agent deployment")
    elif event.component == "dataset":
        action = {"uploaded": "Uploaded", "updated": "Updated", "existing": "Found existing"}.get(event.status)
        if action:
            console.print(f"  {CHECK} {action} dataset '{_SAMPLE_DATASET_FILESET}#{_SAMPLE_DATASET_FILENAME}'")
    elif event.component == "evaluation_config":
        if event.status in {"uploaded", "existing"}:
            action = "Uploaded" if event.status == "uploaded" else "Found existing"
            console.print(
                f"  {CHECK} {action} evaluation config '{_SAMPLE_DATASET_FILESET}#{_SAMPLE_EVAL_CONFIG_FILENAME}'"
            )


# ---------------------------------------------------------------------------
# Interactive flow helpers
# ---------------------------------------------------------------------------


def _select_provider() -> KnownProvider | None:
    """Prompt user to pick one provider. Returns None for 'custom'."""
    options = [(p.name, f"{p.label:<20s} {p.description}") for p in KNOWN_PROVIDERS]
    options.append(("custom", "Custom provider      Enter URL and key manually"))

    result = prompt_choice(
        message="",
        options=options,
        default=KNOWN_PROVIDERS[0].name,
    )

    if result == "custom":
        return None
    return _KNOWN_PROVIDERS_BY_NAME[result]


def _prompt_custom_provider() -> tuple[str, str, str | None]:
    """Prompt for custom provider details. Returns (name, host_url, api_key_or_none)."""
    name = prompt_text(
        "Provider name: ",
        validator=provider_name_validator(),
        hint="Start with a lowercase letter, then lowercase letters, digits, or hyphens; 2-63 chars (e.g. my-vllm-provider)",
    ).strip()

    host_url = prompt_text("Provider base URL: ").strip()
    if not host_url:
        raise typer.Exit(1)

    api_key = prompt_password("API key (leave empty if none): ").strip() or None

    return name, host_url, api_key


def _collect_credential(provider: KnownProvider) -> str:
    """Prompt for the API key, checking the env var first."""
    if not provider.requires_api_key:
        return ""

    env_val = os.environ.get(provider.env_var or "") if provider.env_var else None
    if env_val:
        masked = f"***{env_val[-4:]}" if len(env_val) > 4 else "****"
        console.print(f"  Found {provider.env_var} in environment ({masked})")
        key = prompt_password(f"{provider.label} API key [{masked}]: ")
        return key.strip() if key.strip() else env_val

    key = prompt_password(
        f"{provider.label} API key: ",
        validator=non_empty_validator("API key"),
    )
    return key.strip()


def _reprompt_api_key(label: str) -> str:
    """Ask the user to enter a different API key during interactive validation."""
    key = prompt_password(
        f"{label} API key: ",
        validator=non_empty_validator("API key"),
    )
    return key.strip()


def _confirm_interactive_api_key(
    *,
    provider_name: str,
    host_url: str,
    api_key: str,
    auth_header_format: str | None,
    default_extra_headers: dict[str, str] | None,
    label: str,
) -> str:
    """Validate ``api_key`` interactively; re-prompt or continue on non-valid outcomes.

    Returns the API key to register. Raises ``typer.Exit`` on abort; ``UserCancelled``
    from prompts propagates to the interactive setup handler.
    """
    current_key = api_key
    while True:
        console.print("\n  Validating API key...")
        key_result = _validate_api_key(
            provider_name,
            host_url,
            current_key,
            auth_header_format=auth_header_format,
            default_extra_headers=default_extra_headers,
        )
        if key_result.status == KeyValidationStatus.VALID:
            console.print(f"  {CHECK} API key validated")
            return current_key

        if key_result.status == KeyValidationStatus.REJECTED:
            console.print(f"  {CROSS} {escape(key_result.message)}")
            console.print("  Enter a different API key, or cancel to exit.")
            current_key = _reprompt_api_key(label)
            continue

        console.print(f"  {WARN} {escape(key_result.message)}")
        action = prompt_choice(
            "API key could not be verified. What next?",
            options=_KEY_VALIDATION_ACTIONS,
            default="reenter",
            indent=2,
        )
        if action == "continue":
            return current_key
        if action == "abort":
            raise typer.Exit(1)
        current_key = _reprompt_api_key(label)


def _register_provider_interactive(
    clients: SetupClients,
    *,
    provider_name: str,
    host_url: str,
    api_key: str | None,
    workspace: str,
    auth_header_format: str | None = None,
    default_extra_headers: dict[str, str] | None = None,
) -> None:
    """Create or update secret + provider for idempotent re-runs."""
    secret_name: str | None = None
    if api_key:
        secret_name = f"{provider_name}-api-key"
        if _secret_exists(clients, secret_name, workspace):
            _update_secret(clients, secret_name, api_key, workspace)
            console.print(f"  {CHECK} Updated secret '{secret_name}'")
        else:
            _create_secret(clients, secret_name, api_key, workspace)
            console.print(f"  {CHECK} Created secret '{secret_name}'")

    if _provider_exists(clients, provider_name, workspace):
        _update_provider(
            clients,
            name=provider_name,
            host_url=host_url,
            secret_name=secret_name,
            workspace=workspace,
            auth_header_format=auth_header_format,
            default_extra_headers=default_extra_headers,
        )
        console.print(f"  {CHECK} Updated provider '{provider_name}' ({display_url(host_url)})")
    else:
        _create_provider(
            clients,
            name=provider_name,
            host_url=host_url,
            secret_name=secret_name,
            workspace=workspace,
            auth_header_format=auth_header_format,
            default_extra_headers=default_extra_headers,
        )
        console.print(f"  {CHECK} Registered provider '{provider_name}' ({display_url(host_url)})")


def _probe_response_detail(resp: httpx.Response) -> str:
    """Return a short upstream error string from a probe response, if present."""
    try:
        parsed = resp.json()
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("detail", "title", "message"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:_PROBE_DETAIL_MAX_CHARS]
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()[:_PROBE_DETAIL_MAX_CHARS]
    return ""


def _with_probe_detail(message: str, resp: httpx.Response) -> str:
    detail = _probe_response_detail(resp)
    if not detail or detail in message:
        return message
    return f"{message} {detail}"


def _catalog_model_ids(resp: httpx.Response) -> list[str]:
    """Return model ids from an OpenAI-shaped ``GET /v1/models`` body."""
    try:
        ids: list[str] = []
        for item in resp.json()["data"]:
            model_id = item["id"]
            if isinstance(model_id, str) and model_id.strip():
                ids.append(model_id.strip())
        return ids
    except Exception:
        return []


def _nvidia_build_chat_candidates(model_ids: list[str]) -> list[str]:
    """Return a short list of chat models for key validation (NVIDIA-first, smallest first)."""
    chat_ids = [model_id for model_id in model_ids if _is_usable_chat_model_entity(model_id)]
    ordered = _order_candidates_by_size(chat_ids, largest_first=False)
    return ordered[:_KEY_VALIDATION_CHAT_MAX_ATTEMPTS]


def _inconclusive_message(summary: str, resp: httpx.Response | None = None) -> str:
    """Build fail-open copy that never blames the key for an unverified probe."""
    base = f"Could not verify API key: {summary} {_KEY_UNVERIFIED_SUFFIX}"
    if resp is None:
        return base
    return _with_probe_detail(base, resp)


def _probe_status_result(resp: httpx.Response) -> KeyValidationResult:
    if resp.status_code in _KEY_REJECTED_STATUS_CODES:
        return KeyValidationResult(
            status=KeyValidationStatus.REJECTED,
            message=_with_probe_detail(_KEY_REJECTED_MESSAGE, resp),
        )
    if resp.is_success:
        return KeyValidationResult(status=KeyValidationStatus.VALID)
    return KeyValidationResult(
        status=KeyValidationStatus.INCONCLUSIVE,
        message=_inconclusive_message(f"probe received HTTP {resp.status_code}.", resp),
    )


def _nvidia_build_chat_probe(
    host_url: str,
    headers: dict[str, str],
    catalog_resp: httpx.Response,
    *,
    timeout: float,
) -> KeyValidationResult:
    """POST chat completions to catalog candidates until auth is proven or exhausted."""
    candidates = _nvidia_build_chat_candidates(_catalog_model_ids(catalog_resp))
    if not candidates:
        return KeyValidationResult(
            status=KeyValidationStatus.INCONCLUSIVE,
            message=_inconclusive_message("provider catalog listed no usable chat models."),
        )

    chat_url = f"{host_url.rstrip('/')}/v1/chat/completions"
    last_warning = _inconclusive_message("probe did not find a callable chat model.")
    for model_id in candidates:
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Respond with 'OK'"}],
            "max_tokens": 1,
        }
        try:
            chat_resp = httpx.request("POST", chat_url, headers=headers, json=body, timeout=timeout)
        except httpx.TimeoutException:
            logger.debug("NVIDIA Build chat probe timed out for '%s'", model_id)
            last_warning = _inconclusive_message("provider probe timed out.")
            continue
        except Exception as exc:
            logger.debug("NVIDIA Build chat probe failed for '%s': %s", model_id, exc)
            last_warning = _inconclusive_message(f"provider probe failed ({exc}).")
            continue
        if chat_resp.status_code in _KEY_REJECTED_STATUS_CODES:
            return _probe_status_result(chat_resp)
        if chat_resp.is_success:
            return KeyValidationResult(status=KeyValidationStatus.VALID)
        last_warning = _probe_status_result(chat_resp).message
    return KeyValidationResult(status=KeyValidationStatus.INCONCLUSIVE, message=last_warning)


def _validate_api_key(
    provider_name: str,
    host_url: str,
    api_key: str | None,
    *,
    auth_header_format: str | None = None,
    default_extra_headers: dict[str, str] | None = None,
    timeout: float = _KEY_VALIDATION_TIMEOUT,
) -> KeyValidationResult:
    """Probe the provider with the API key to detect auth failures early.

    Makes a lightweight request to an auth-required endpoint.
    Returns ``REJECTED`` only on a definitive 401/403 credential rejection.
    Unavailable probe targets, other HTTP statuses, network errors, and unknown
    providers are ``INCONCLUSIVE`` (fail-open) so setup can still register the
    provider after an explicit warning.
    """
    if not api_key:
        return KeyValidationResult(status=KeyValidationStatus.VALID)

    probe = _PROBE_CONFIGS.get(provider_name)
    if probe is None:
        return KeyValidationResult(
            status=KeyValidationStatus.INCONCLUSIVE,
            message=(
                f"Could not verify API key: no validation probe configured for "
                f"provider '{provider_name}'. {_KEY_UNVERIFIED_SUFFIX}"
            ),
        )

    headers: dict[str, str] = {}
    if auth_header_format:
        header_name, _, template = auth_header_format.partition(":")
        headers[header_name.strip()] = template.strip().replace("{{ auth_secret }}", api_key)
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    if default_extra_headers:
        headers.update(default_extra_headers)

    url = f"{host_url.rstrip('/')}/{probe.path}"

    try:
        resp = httpx.request(
            probe.method,
            url,
            headers=headers,
            json=probe.body,
            timeout=timeout,
        )
        if resp.status_code in _KEY_REJECTED_STATUS_CODES:
            return _probe_status_result(resp)
        if resp.is_success:
            if provider_name == "nvidia-build":
                return _nvidia_build_chat_probe(host_url, headers, resp, timeout=timeout)
            return KeyValidationResult(status=KeyValidationStatus.VALID)
        return _probe_status_result(resp)
    except httpx.TimeoutException:
        logger.debug("API key validation timed out for '%s'", provider_name)
        return KeyValidationResult(
            status=KeyValidationStatus.INCONCLUSIVE,
            message=_inconclusive_message("provider probe timed out."),
        )
    except Exception as exc:
        logger.debug("API key validation failed for '%s': %s", provider_name, exc)
        return KeyValidationResult(
            status=KeyValidationStatus.INCONCLUSIVE,
            message=_inconclusive_message(f"provider probe failed ({exc})."),
        )


def _select_model_pair(
    clients: SetupClients,
    workspace: str,
    *,
    provider_name: str | None = None,
) -> ModelPair | None:
    """Let the user pick default and fast models from one provider."""
    display_models = _get_all_model_choices(clients, workspace, provider_name=provider_name)
    if not display_models:
        console.print(f"  {WARN} No models discovered yet. You can select models later.")
        return None

    entity_ids = [entity_id for entity_id, _ in display_models]
    if _pick_default_chat_entity(entity_ids) is None:
        console.print(f"  {WARN} No usable chat models discovered yet. You can select models later.")
        return None

    suggested = _select_usable_model_pair(clients, workspace, entity_ids)
    if suggested is None:
        console.print(f"  {WARN} None of the discovered models served a test request; choose a model explicitly.")

    default_model = prompt_search_select(
        "Choose your default model (used for quality-critical agent work):",
        choices=display_models,
        default=suggested.default if suggested else None,
        hint="Press Enter to accept the default, or type to search."
        if suggested
        else "Type to search models you can access.",
    )
    fast_default = suggested.fast if suggested else default_model
    fast_hint = (
        "Press Enter to reuse the default model, or type to search."
        if fast_default == default_model
        else "Press Enter to accept the suggested fast model, or type to search."
    )
    fast = prompt_search_select(
        "Choose your fast model (used for latency-sensitive agent work):",
        choices=display_models,
        default=fast_default,
        hint=fast_hint,
    )
    return ModelPair(default=default_model, fast=fast)


def _save_model_pair(cli_context: CLIContext, model_pair: ModelPair) -> None:
    """Persist the default quality model and the low-latency model."""
    context = cli_context.get_sdk_context()
    params: ConfigParams = {
        "default_model": model_pair.default,
        "fast_model": model_pair.fast,
    }
    Config.write(params, context_name=context.context_name)


def _check_ollama_running(host_url: str) -> bool:
    """Probe Ollama endpoint to check if it's running."""
    try:
        resp = httpx.get(f"{host_url.rstrip('/')}/models", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Auto (non-interactive) mode
# ---------------------------------------------------------------------------


def _auto_setup(clients: SetupClients, workspace: str, inference_base_url: str | None = None) -> str | None:
    """Register a provider from environment variables and return its name."""
    override_base_url = inference_base_url.strip() if inference_base_url else ""
    for key_var, url_var in _AUTO_ENV_VARS:
        api_key = os.environ.get(key_var)
        if not api_key:
            continue

        base_url = override_base_url or (os.environ.get(url_var, "").strip() if url_var else "")
        env_provider = next((p for p in KNOWN_PROVIDERS if p.env_var == key_var and p.requires_api_key), None)
        if base_url:
            known = _resolve_provider_for_url(base_url)
            if known:
                provider_name = known.name
            else:
                hostname = urlparse(base_url).hostname or "custom"
                provider_name = hostname.replace(".", "-")
            host_url = base_url
            auth_provider = known or env_provider
            auth_header_format = auth_provider.auth_header_format if auth_provider else None
            default_extra_headers = auth_provider.default_extra_headers if auth_provider else None
        else:
            if env_provider is None:
                continue
            provider_name = env_provider.name
            host_url = env_provider.host_url
            auth_header_format = env_provider.auth_header_format
            default_extra_headers = env_provider.default_extra_headers

        key_result = _validate_api_key(
            provider_name,
            host_url,
            api_key,
            auth_header_format=auth_header_format,
            default_extra_headers=default_extra_headers,
        )
        if key_result.status == KeyValidationStatus.REJECTED:
            console.print(f"  {CROSS} {escape(key_result.message)}")
            console.print(f"  Check the value of ${key_var} and try again.")
            raise typer.Exit(1)
        if key_result.status == KeyValidationStatus.INCONCLUSIVE:
            console.print(f"  {WARN} {escape(key_result.message)}")

        secret_name = f"{provider_name}-api-key"
        if _secret_exists(clients, secret_name, workspace):
            _update_secret(clients, secret_name, api_key, workspace)
            console.print(f"  {CHECK} Updated secret '{secret_name}' (from ${key_var})")
        else:
            _create_secret(clients, secret_name, api_key, workspace)
            console.print(f"  {CHECK} Created secret '{secret_name}' (from ${key_var})")

        if _provider_exists(clients, provider_name, workspace):
            _update_provider(
                clients,
                name=provider_name,
                host_url=host_url,
                secret_name=secret_name,
                workspace=workspace,
                auth_header_format=auth_header_format,
                default_extra_headers=default_extra_headers,
            )
            console.print(f"  {CHECK} Updated provider '{provider_name}' ({display_url(host_url)})")
        else:
            _create_provider(
                clients,
                name=provider_name,
                host_url=host_url,
                secret_name=secret_name,
                workspace=workspace,
                auth_header_format=auth_header_format,
                default_extra_headers=default_extra_headers,
            )
            console.print(f"  {CHECK} Registered provider '{provider_name}' ({display_url(host_url)})")

        return provider_name

    return None


def _require_supported_python() -> None:
    """Exit if this interpreter is outside the supported Python versions."""
    v = sys.version_info[:2]
    v_min, v_max = _SUPPORTED_PYTHON_MIN, _SUPPORTED_PYTHON_MAX
    if v_min <= v <= v_max:
        return

    running = ".".join(str(part) for part in sys.version_info[:3])
    console.print(f"\n{CROSS} Unsupported Python {running}.")
    console.print(f"  NeMo Helix requires Python {v_min[0]}.{v_min[1]}-{v_max[0]}.{v_max[1]}.")
    console.print("  Reinstall the CLI against a supported interpreter:")
    console.print(f'    [cyan]uv tool install --python {v_max[0]}.{v_max[1]} --reinstall "nemo-helix[all]"[/cyan]')
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@handle_errors
def setup_command(
    ctx: typer.Context,
    auto: Annotated[
        bool,
        typer.Option("--auto", help="Non-interactive mode: register provider from environment variables"),
    ] = False,
    workspace: Annotated[
        str | None,
        typer.Option("--workspace", "-w", help=WORKSPACE_HELP),
    ] = None,
    start_services: Annotated[
        bool | None,
        typer.Option("--start-services/--no-start-services", help="Start local Helix services"),
    ] = None,
    install_skills: Annotated[
        bool | None,
        typer.Option("--install-skills/--no-install-skills", help="Install NeMo skills for coding agents"),
    ] = None,
    skills_agents: Annotated[
        str | None,
        typer.Option(
            "--skills-agents",
            help=(
                "Comma-separated list of agents to install skills for (e.g. 'codex,cursor'). "
                "Default: all detected. Only applied when --install-skills is set."
            ),
        ),
    ] = None,
    skills_scope: Annotated[
        Scope | None,
        typer.Option(
            "--skills-scope",
            help=(
                "Install scope for skills: 'project' (this repo) or 'user' (home). "
                "Default: project. Only applied when --install-skills is set."
            ),
            case_sensitive=False,
        ),
    ] = None,
    skills_from: Annotated[
        str | None,
        typer.Option(
            "--skills-from",
            help=(
                "Comma-separated list of skill sources to install from "
                "(e.g. 'nemo-helix,nemo-evals-plugin'). Use 'nemo-helix' "
                "for the built-in set. Default: all sources. "
                "Only applied when --install-skills is set."
            ),
        ),
    ] = None,
    skills_path: Annotated[
        Path | None,
        typer.Option(
            "--skills-path",
            help="Install skills to a custom directory for another coding agent",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Retry an interrupted setup using the normal idempotent setup path.",
        ),
    ] = False,
    ready_timeout: Annotated[
        int | None,
        typer.Option(
            "--ready-timeout",
            help=f"Seconds to wait for platform readiness (default: {_SERVICE_STARTUP_TIMEOUT_SECONDS})",
        ),
    ] = None,
    inference_base_url: Annotated[
        str | None,
        typer.Option(
            "--inference-base-url",
            help=("Base URL for the provider registered by --auto. Equivalent to NEMO_DEFAULT_INFERENCE_BASE_URL."),
        ),
    ] = None,
) -> None:
    """Set up NeMo Helix: connect or start services, configure a provider, install skills.

    Uses an already-running platform, starts local services, or connects the
    CLI to an existing remote deployment. Then selects and registers an
    inference provider, picks default and fast agent models, and installs
    coding agent skills. Interactive mode can create a sample workspace and
    Fabric email security agent.

    The active config context remembers the Platform URL. When a remote
    deployment is already reachable, setup asks whether to continue with it,
    start local services instead, or connect to a different remote URL.

    To override the URL for one run only:
      nemo --base-url http://localhost:8080 setup

    To persist a different URL:
      nemo config set --base-url http://localhost:8080

    Requires an interactive terminal (TTY). In non-interactive contexts
    (CI, piped input), pass --auto to use environment variables instead.

    Use --auto for non-interactive setup from environment variables
    (NEMO_DEFAULT_INFERENCE_KEY with optional NEMO_DEFAULT_INFERENCE_BASE_URL,
    NVIDIA_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY).
    Override the provider URL with --inference-base-url. Override the selected
    pair with NEMO_DEFAULT_MODEL and NEMO_FAST_MODEL.

    Examples:
      nemo setup
      nemo setup --auto
      nemo setup --auto --start-services --install-skills
      nemo setup --auto --start-services --ready-timeout 360
      nemo setup --auto --start-services --inference-base-url https://inference-api.nvidia.com/v1
      NHX_BASE_URL=https://nhx.example.com NHX_ACCESS_TOKEN=... nemo setup --auto --no-start-services
      nemo setup --workspace my-workspace
      nemo setup --no-install-skills
      nemo --base-url http://localhost:8080 setup
    """
    cli_context: CLIContext = ctx.obj
    base_url = cli_context.get_base_url() or DEFAULT_BASE_URL

    _require_supported_python()

    console.print("\n[bold cyan]NeMo Helix Setup[/bold cyan]\n")
    _print_legacy_directory_notice()
    if resume:
        console.print(f"{CHECK} Retrying setup using the normal idempotent setup path.\n")

    if not auto and not is_interactive():
        console.print(f"\n{CROSS} Detected non-interactive shell. Pass [bold]--auto[/bold] or run in a TTY.\n")
        raise typer.Exit(1)

    effective_timeout = _SERVICE_STARTUP_TIMEOUT_SECONDS if ready_timeout is None else ready_timeout
    if effective_timeout <= 0:
        raise typer.BadParameter("--ready-timeout must be greater than 0", param_hint="--ready-timeout")
    certificate_authority = cli_context.get_sdk_context().cluster.certificate_authority
    # Resolve once, before any path reads it. Every branch below either writes
    # a context seeded from this value or provisions into it, so resolving here
    # (against the context the user is currently on) is both the simplest and
    # the only non-circular ordering.
    workspace = _resolve_setup_workspace(cli_context, workspace)
    try:
        configured_base_url = base_url
        service_result = _maybe_start_services(
            base_url,
            auto,
            start_services,
            timeout=effective_timeout,
            certificate_authority=certificate_authority,
        )
        if service_result == "start_local":
            _configure_local_connection(cli_context, workspace)
            base_url = DEFAULT_BASE_URL
            certificate_authority = cli_context.get_sdk_context().cluster.certificate_authority
            service_result = _maybe_start_services(
                base_url,
                auto,
                start_services=True,
                timeout=effective_timeout,
                certificate_authority=certificate_authority,
            )
        if service_result == "connect_remote":
            base_url = _prompt_remote_base_url(
                default_url=configured_base_url,
                certificate_authority=certificate_authority,
            )
            _bootstrap_config_if_missing(base_url, workspace)
            cli_context.reset_sdk_context()
            _configure_remote_connection(cli_context, base_url, workspace)
            _ensure_platform_auth(cli_context)
            certificate_authority = cli_context.get_sdk_context().cluster.certificate_authority
    except UserCancelled:
        console.print(f"\n{WARN} Setup cancelled.")
        raise typer.Exit(0) from None

    if not _check_platform_reachable_with_retries(base_url, certificate_authority=certificate_authority):
        console.print(f"\n{CROSS} Cannot reach platform at {display_url(base_url)}")
        raise typer.Exit(1)

    console.print(f"{CHECK} Platform reachable at {display_url(base_url)}\n")

    # Ensure the config file exists on disk so later Config.write() calls
    # (e.g. saving the default model) can find the cluster and context.
    # Without this, a fresh install (no config.yaml) hits "Cluster
    # 'default-cluster' does not exist" when _save_model_pair runs.
    _bootstrap_config_if_missing(base_url, workspace)
    cli_context.reset_sdk_context()
    certificate_authority = cli_context.get_sdk_context().cluster.certificate_authority

    workspaces = cli_context.typed_client(WorkspacesClient)

    try:
        workspaces.get_workspace(name=workspace).data()
    except Exception:
        try:
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
            console.print(f"  {CHECK} Created workspace '{workspace}'")
        except Exception as create_err:
            # Distinguish a race (workspace appeared between retrieve and create)
            # from a real failure (permissions, server error).
            try:
                workspaces.get_workspace(name=workspace).data()
            except Exception:
                raise create_err from None

    skills_agents_list = _parse_csv_flag(skills_agents)
    skills_from_list = _parse_csv_flag(skills_from)
    clients = SetupClients.from_context(cli_context)

    try:
        if auto:
            _run_auto_mode(
                cli_context,
                clients,
                workspace,
                base_url,
                install_skills,
                skills_agents=skills_agents_list,
                skills_scope=skills_scope,
                skills_from=skills_from_list,
                skills_path=skills_path,
                certificate_authority=certificate_authority,
                inference_base_url=inference_base_url,
            )
        else:
            _run_interactive_mode(
                cli_context,
                clients,
                workspace,
                base_url,
                install_skills,
                skills_agents=skills_agents_list,
                skills_scope=skills_scope,
                skills_from=skills_from_list,
                skills_path=skills_path,
                certificate_authority=certificate_authority,
            )
    except typer.Exit as exc:
        # A clean user-cancel raises typer.Exit(0); that is a normal end of the
        # flow, not a failure, so it must not corrupt the onboarding funnel.
        # Only a non-zero exit code counts as ERROR. Re-raise unchanged either way.
        status = TaskStatusEnum.COMPLETED if exc.exit_code == 0 else TaskStatusEnum.ERROR
        emit.emit_event(OnboardingStepEvent(step="setup_finished", task_status=status))
        raise
    except Exception:
        # Any real (non-Exit) failure: setup did not finish cleanly.
        emit.emit_event(OnboardingStepEvent(step="setup_finished", task_status=TaskStatusEnum.ERROR))
        raise
    emit.emit_event(OnboardingStepEvent(step="setup_finished", task_status=TaskStatusEnum.COMPLETED))


def _run_auto_mode(
    cli_context: CLIContext,
    clients: SetupClients,
    workspace: str,
    base_url: str,
    install_skills: bool | None,
    *,
    skills_agents: list[str] | None = None,
    skills_scope: Scope | None = None,
    skills_from: list[str] | None = None,
    skills_path: Path | None = None,
    certificate_authority: str | None = None,
    inference_base_url: str | None = None,
) -> None:
    """Non-interactive provider registration from environment variables."""
    console.print("[bold]Auto-detecting provider from environment...[/bold]\n")
    provider_name = _auto_setup(clients, workspace, inference_base_url=inference_base_url)
    if provider_name is None:
        console.print(f"{CROSS} No provider credentials found in environment.")
        env_var_names = ", ".join(key for key, _ in _AUTO_ENV_VARS)
        console.print(f"  Set one of: {env_var_names}")
        raise typer.Exit(1)

    console.print("\n  Waiting for model discovery...")
    entity_ids: list[str] = []
    for attempt in range(_MODEL_DISCOVERY_MAX_ROUNDS):
        deadline = time.time() + _MODEL_DISCOVERY_ROUND_SECONDS
        while time.time() < deadline:
            entity_ids = _get_all_model_entity_ids(clients, workspace, provider_name=provider_name)
            if entity_ids:
                break
            _pause(_MODEL_DISCOVERY_POLL_INTERVAL)
        if entity_ids:
            break
        if attempt < _MODEL_DISCOVERY_MAX_ROUNDS - 1:
            console.print(f"  {WARN} Models not available yet, retrying...")

    # Auto mode discovers models via an inline loop rather than _wait_for_models,
    # so emit the models_discovered event here to cover the non-interactive path.
    emit.emit_event(
        OnboardingStepEvent(
            step="models_discovered",
            task_status=TaskStatusEnum.COMPLETED,
            models_discovered_bucket=_bucket_model_count(len(entity_ids)),
        )
    )

    # An explicit NEMO_DEFAULT_MODEL is taken as given; anything auto-selected
    # has to answer a real inference request first.
    default_override = os.environ.get("NEMO_DEFAULT_MODEL", "").strip()
    fast_override = os.environ.get("NEMO_FAST_MODEL", "").strip()
    if default_override:
        model_pair = ModelPair(default=default_override, fast=fast_override or default_override)
    elif entity_ids:
        selected = _select_usable_model_pair(clients, workspace, entity_ids)
        model_pair = ModelPair(default=selected.default, fast=fast_override or selected.fast) if selected else None
    else:
        model_pair = None

    if model_pair:
        _save_model_pair(cli_context, model_pair)
        console.print(f"  {CHECK} Default model: {model_pair.default}")
        console.print(f"  {CHECK} Fast model: {model_pair.fast}")
    else:
        if fast_override:
            console.print(f"  {WARN} NEMO_FAST_MODEL is ignored until a default model is available")
        if entity_ids:
            console.print(f"  {WARN} No default model set (none of the discovered models served a test request)")
            console.print("  The provider advertises models this account cannot run. Check the account's")
            console.print("  model entitlements, or pick a model yourself:")
        else:
            console.print(f"  {WARN} No default model set (no models discovered yet)")
            console.print("  Run [cyan]nemo setup[/cyan] again after models sync, or set the models via env vars:")
        console.print("    [cyan]export NEMO_DEFAULT_MODEL=<model>[/cyan]")
        console.print("    [cyan]export NEMO_FAST_MODEL=<model>[/cyan]")

    _maybe_install_skills(
        auto=True,
        install_skills=install_skills,
        skills_agents=skills_agents,
        skills_scope=skills_scope,
        skills_from=skills_from,
        skills_path=skills_path,
    )
    if not _verify_platform_health(base_url, certificate_authority=certificate_authority):
        raise typer.Exit(1)

    if model_pair:
        console.print(f"\n{CHECK} [green]Setup complete![/green]")
    else:
        console.print(f"\n{WARN} [yellow]Setup complete with warnings.[/yellow]")
        console.print("  No default model was set; review the warnings above before using the platform.")


def _run_interactive_mode(
    cli_context: CLIContext,
    clients: SetupClients,
    workspace: str,
    base_url: str,
    install_skills: bool | None,
    *,
    skills_agents: list[str] | None = None,
    skills_scope: Scope | None = None,
    skills_from: list[str] | None = None,
    skills_path: Path | None = None,
    certificate_authority: str | None = None,
) -> str:
    """Walk the user through provider selection, credential entry, and model choice."""
    try:
        provider_name, host_url, api_key, auth_header_format, default_extra_headers = _interactive_collect_provider()

        if api_key:
            known = _KNOWN_PROVIDERS_BY_NAME.get(provider_name)
            label = known.label if known is not None else provider_name
            api_key = _confirm_interactive_api_key(
                provider_name=provider_name,
                host_url=host_url,
                api_key=api_key,
                auth_header_format=auth_header_format,
                default_extra_headers=default_extra_headers,
                label=label,
            )

        console.print("\n[bold]Step 3: Register model provider[/bold]\n")
        _register_provider_interactive(
            clients,
            provider_name=provider_name,
            host_url=host_url,
            api_key=api_key,
            workspace=workspace,
            auth_header_format=auth_header_format,
            default_extra_headers=default_extra_headers,
        )

        console.print("\n[bold]Step 4: Discover models[/bold]\n")
        console.print("  Waiting for model discovery...")
        models = _wait_for_models(clients, provider_name, workspace, host_url=host_url)
        if models:
            console.print(f"  {CHECK} Found {len(models)} model(s)")
        else:
            console.print(f"  {WARN} No models discovered yet (provider may still be syncing)")

        console.print("\n[bold]Step 5: Choose agent models[/bold]\n")
        fallback_model_choices = _get_all_model_choices(clients, workspace) if not models else []
        if not models and fallback_model_choices:
            console.print(f"  {WARN} Models from existing providers are available, but not from '{provider_name}' yet.")

        model_pair = _select_model_pair(clients, workspace, provider_name=provider_name) if models else None
        default_model = model_pair.default if model_pair else None
        if model_pair:
            _save_model_pair(cli_context, model_pair)
            console.print(f"  {CHECK} Default model set to {_display_model_name(model_pair.default)}")
            console.print(f"  {CHECK} Fast model set to {_display_model_name(model_pair.fast)}")
        else:
            console.print(f"  {WARN} No agent models set for this provider yet.")
            console.print("  Run [cyan]nemo setup[/cyan] again after models sync")

        console.print("\n[bold]Step 6: Install skills[/bold]\n")
        _maybe_install_skills(
            auto=False,
            install_skills=install_skills,
            skills_agents=skills_agents,
            skills_scope=skills_scope,
            skills_from=skills_from,
            skills_path=skills_path,
        )

        _print_setup_complete(
            base_url,
            provider_name,
            default_model,
            fast_model=model_pair.fast if model_pair else None,
            certificate_authority=certificate_authority,
        )

        selected_path = _prompt_post_setup_path()
        if selected_path == "sample":
            if not default_model:
                console.print(f"  {WARN} No default model selected, skipping sample agent setup")
                return selected_path

            agents_client = cli_context.typed_client(AgentsClient)
            progress_workspace = None
            deployment_submitted = False
            try:
                sample = None
                with agents_client.create_sample_agent(
                    body=CreateSampleAgentRequest(model=default_model)
                ).stream() as events:
                    for event in events:
                        if event.kind == "progress":
                            _print_sample_progress(event)
                            progress_workspace = event.workspace or progress_workspace
                            deployment_submitted |= event.component == "deployment" and event.status == "submitted"
                        elif event.kind == "error":
                            console.print(f"  {WARN} {event.message or 'Sample agent setup failed'}")
                            if event.workspace:
                                _print_sample_setup_complete(base_url, event.workspace, complete=False)
                            return selected_path
                        elif event.kind == "done":
                            sample = event.result
            except NemoHTTPError as exc:
                detail = exc.body.get("detail") if isinstance(exc.body, dict) else None
                if isinstance(detail, dict):
                    workspace_name = detail.get("workspace")
                    if isinstance(workspace_name, str):
                        console.print(
                            f"  {WARN} Sample agent setup failed at {detail.get('failed_step', 'provisioning')}"
                        )
                        _print_sample_setup_complete(base_url, workspace_name, complete=False)
                        return selected_path
                console.print(f"  {WARN} Could not create sample agent: {exc.detail}")
                return selected_path
            except Exception as exc:
                console.print(f"  {WARN} Could not create sample agent: {exc}")
                if progress_workspace:
                    _print_sample_setup_complete(base_url, progress_workspace, complete=False)
                return selected_path

            if sample is None:
                console.print(f"  {WARN} Sample agent setup ended without a completion event")
                if progress_workspace:
                    _print_sample_setup_complete(base_url, progress_workspace, complete=False)
                return selected_path
            agent_ready = _wait_for_sample_deployment(agents_client, sample, submitted=deployment_submitted)
            emit.emit_event(
                OnboardingStepEvent(
                    step="agent_deployed",
                    task_status=TaskStatusEnum.COMPLETED if agent_ready else TaskStatusEnum.ERROR,
                    agent_deployed=agent_ready,
                )
            )
            _print_sample_setup_complete(base_url, sample.workspace, complete=agent_ready)
        return selected_path

    except UserCancelled:
        console.print(f"\n{WARN} Setup cancelled.")
        raise typer.Exit(0) from None


def _interactive_collect_provider() -> tuple[str, str, str | None, str | None, dict[str, str] | None]:
    """Steps 1-2: select provider and collect credentials.

    Returns (provider_name, host_url, api_key, auth_header_format, default_extra_headers).
    """
    console.print("[bold]Step 1: Choose a model provider[/bold]\n")
    selected = _select_provider()

    if selected is None:
        name, host_url, api_key = _prompt_custom_provider()
        return name, host_url, api_key, None, None

    if selected.name == "ollama":
        if not _check_ollama_running(selected.host_url):
            console.print(f"\n  {WARN} Ollama does not appear to be running at {selected.host_url}")
            console.print("  Make sure Ollama is started before using it for inference.")
        else:
            console.print(f"\n  {CHECK} Ollama detected at {selected.host_url}")

    if selected.requires_api_key:
        console.print("\n[bold]Step 2: Enter API key[/bold]\n")
        api_key: str | None = _collect_credential(selected)
    else:
        api_key = None

    return selected.name, selected.host_url, api_key, selected.auth_header_format, selected.default_extra_headers


def _print_setup_complete(
    base_url: str,
    provider_name: str,
    default_model: str | None,
    *,
    fast_model: str | None = None,
    certificate_authority: str | None = None,
) -> None:
    """Verify platform health and print the setup summary."""
    if not _verify_platform_health(base_url, certificate_authority=certificate_authority):
        raise typer.Exit(1)

    lines = [
        f"[bold]Platform:[/bold] {display_url(base_url)}",
        f"[bold]Provider:[/bold] {provider_name}",
    ]
    if default_model:
        lines.append(f"[bold]Default model:[/bold] {_display_model_name(default_model)}")
    if fast_model:
        lines.append(f"[bold]Fast model:[/bold] {_display_model_name(fast_model)}")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Setup complete[/bold]",
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 1),
        )
    )


def _prompt_post_setup_path() -> str:
    """Prompt for the next path without starting it."""
    return prompt_choice(
        "How would you like to get started?",
        _POST_SETUP_OPTIONS,
        default="sample",
        indent=2,
    )
