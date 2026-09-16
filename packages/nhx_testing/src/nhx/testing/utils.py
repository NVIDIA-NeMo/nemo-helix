# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Testing utility functions for NeMo Helix services."""

import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nemo_helix import ConflictError, NeMoHelix, NotFoundError, omit
from nemo_helix.types.inference import ModelProvider, ServedModelMapping
from nemo_helix.types.workspaces import WorkspaceMember
from nhx.common.entities.constants import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nhx.common.entities.utils import get_random_id

NemoRun = Callable[..., subprocess.CompletedProcess[str]]

_E2E_IGW_WAIT_TIMEOUT_SEC = 60

_ENTITY_NAME_PATTERN = re.compile(NAME_PATTERN)


def _repo_venv_nemo_command(repo_root: Path) -> list[str] | None:
    """Return the current repo venv's nemo console script when pytest already runs inside it."""
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return None

    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    suffixes = (".exe", "") if os.name == "nt" else ("",)
    for suffix in suffixes:
        candidate = Path(venv) / scripts_dir / f"nemo{suffix}"
        if not candidate.exists():
            continue
        try:
            candidate.resolve().relative_to(repo_root.resolve())
        except ValueError:
            continue
        return [str(candidate)]
    return None


def _nemo_local_command(repo_root: Path) -> list[str]:
    return _repo_venv_nemo_command(repo_root) or ["uv", "run", "--project", str(repo_root), "--frozen", "nemo"]


def assert_exit_0(result: subprocess.CompletedProcess[str], msg: str) -> subprocess.CompletedProcess[str]:
    """Assert that a CLI invocation succeeded, including output in the failure message."""
    assert result.returncode == 0, f"{msg}: {result.stderr or result.stdout}"
    return result


def get_repo_root() -> Path:
    """Return the repository root using git."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def run_nemo_local(
    *args: str,
    base_url: str | None = None,
    workspace: str | None = None,
    env_extra: dict[str, str] | None = None,
    timeout: int = 120,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the NeMo CLI (``nemo``) from repo root.

    Used by tests that target local CLI behavior as well as E2E flows that
    need to point the CLI at a specific platform instance.

    Pass ``base_url`` and ``workspace`` to inject ``NHX_BASE_URL`` and
    ``NHX_WORKSPACE`` directly. The CLI gets an empty temporary
    ``NHX_CONFIG_FILE`` so test calls do not read a developer's local CLI/SDK
    config. Pass ``cwd`` to run from a different directory (e.g. a ``tmp_path``
    with a fake ``.git`` marker so ``skills install`` writes there instead of
    the real repo root).
    """
    repo_root = get_repo_root()
    with tempfile.TemporaryDirectory(prefix="nhx-cli-config-") as config_dir:
        env = os.environ.copy()
        config_path = Path(config_dir) / "config.yaml"
        config_path.write_text("{}\n")
        env["NHX_CONFIG_FILE"] = str(config_path)
        if not env_extra or "NEMO_TELEMETRY_ENABLED" not in env_extra:
            env["NEMO_TELEMETRY_ENABLED"] = "false"
        if base_url is not None:
            env["NHX_BASE_URL"] = base_url.rstrip("/")
        if workspace is not None:
            env["NHX_WORKSPACE"] = workspace
        if env_extra:
            env.update(env_extra)
        cmd = [*_nemo_local_command(repo_root), *args]
        return subprocess.run(
            cmd,
            cwd=cwd or repo_root,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )


@dataclass
class MockProviderResponse:
    """Convenience wrapper to configure dynamic mock LLM responses with status codes."""

    response_code: int = 200
    response_body: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"response_code": self.response_code, "response_body": self.response_body}


def _serialize_mock_response_map(
    response_map: dict[str, list[MockProviderResponse]],
) -> str:
    serialized: dict[str, list[dict[str, Any]]] = {}
    for model, responses in response_map.items():
        serialized[model] = [response.to_json() for response in responses]
    return json.dumps(serialized)


def wait_for_model_entity(
    sdk: NeMoHelix,
    workspace: str,
    model_name: str,
    timeout: float = _E2E_IGW_WAIT_TIMEOUT_SEC,
    poll_interval: float = 0.5,
    ensure_virtual_model: bool = False,
    should_autoprovision_virtual_model: bool = True,
) -> None:
    """Poll until a model entity is available in IGW's model cache.

    Uses the OpenAI ``GET /v1/models/{name}`` route, which reads
    :attr:`~nhx.core.inference_gateway.api.model_cache.ModelCache.model_entity_info_map`
    and does **not** require a VirtualModel. Do not poll the model-entity proxy route
    here — that route resolves a VirtualModel first and will 404 until IGW's separate
    VirtualModel cache refreshes, even when the model entity is already served.

    Useful in E2E tests that create a mock provider to wait for the model cache to refresh.

    Args:
        sdk: The NeMoHelix SDK client.
        workspace: The workspace containing the model entity.
        model_name: The model entity name (without workspace prefix).
        timeout: Maximum time to wait in seconds (default: 60).
        poll_interval: Time between polls in seconds (default: 0.5).
        ensure_virtual_model: Recreate the passthrough VirtualModel before
            each poll. Useful for E2E tests where controller cleanup and IGW
            cache refresh can race helper-created VMs.
        should_autoprovision_virtual_model: Whether VirtualModels created by
            ``ensure_virtual_model`` should be marked controller-managed.

    Raises:
        TimeoutError: If the model entity is not available within the timeout.
    """
    start = time.time()
    last_error: Exception | None = None

    while time.time() - start < timeout:
        try:
            if ensure_virtual_model:
                _create_passthrough_virtual_model_once(
                    sdk,
                    workspace,
                    model_name,
                    autoprovisioned=should_autoprovision_virtual_model,
                )
            sdk.inference.gateway.model.get(
                "v1/models",
                name=model_name,
                workspace=workspace,
            )
            return
        except NotFoundError as e:
            # Continue polling if model is not found yet
            last_error = e
            time.sleep(poll_interval)
            continue
        except Exception:
            raise

    raise TimeoutError(
        f"Model entity {workspace}/{model_name} not available after {timeout}s. Last error: {last_error}"
    )


def wait_for_virtual_model(
    sdk: NeMoHelix,
    workspace: str,
    name: str,
    timeout: float = _E2E_IGW_WAIT_TIMEOUT_SEC,
    poll_interval: float = 0.5,
) -> None:
    """Poll until a VirtualModel exists in the entity store (platform SDK).

    This confirms the VirtualModel document was persisted. It does **not** mean
    IGW's in-process VirtualModel cache has refreshed yet — use
    :func:`wait_for_igw_virtual_model` before hitting model-entity or OpenAI
    inference proxy routes.

    Args:
        sdk: The NeMoHelix SDK client.
        workspace: The workspace containing the VirtualModel.
        name: The VirtualModel name (without workspace prefix). For an
            autoprovisioned passthrough VM, this is the served model entity name.
        timeout: Maximum time to wait in seconds (default: 60).
        poll_interval: Time between polls in seconds (default: 0.5).

    Raises:
        TimeoutError: If the VirtualModel is not available within the timeout.
    """
    start = time.time()
    last_error: Exception | None = None

    while time.time() - start < timeout:
        try:
            sdk.inference.virtual_models.retrieve(name=name, workspace=workspace)
            return
        except NotFoundError as e:
            last_error = e
            time.sleep(poll_interval)
            continue
        except Exception:
            raise

    raise TimeoutError(f"VirtualModel {workspace}/{name} not available after {timeout}s. Last error: {last_error}")


def _create_passthrough_virtual_model_once(
    sdk: NeMoHelix,
    workspace: str,
    name: str,
    *,
    autoprovisioned: bool = True,
) -> None:
    try:
        sdk.inference.virtual_models.create(
            workspace=workspace,
            name=name,
            default_model_entity=f"{workspace}/{name}",
            autoprovisioned=autoprovisioned,
        )
    except ConflictError:
        pass


def ensure_passthrough_virtual_model(
    sdk: NeMoHelix,
    workspace: str,
    name: str,
    timeout: float = 20,
    poll_interval: float = 0.5,
    autoprovisioned: bool = True,
) -> None:
    """Create or confirm a passthrough VirtualModel.

    E2E tests can race the models controller: the controller may rediscover a
    mock provider and briefly delete a helper-created VM before the test starts
    routing by that entity name. Retrying create+retrieve makes the helper
    converge on the entity-specific VM the test requested.
    """
    start = time.time()
    last_error: Exception | None = None

    while time.time() - start < timeout:
        try:
            _create_passthrough_virtual_model_once(
                sdk,
                workspace,
                name,
                autoprovisioned=autoprovisioned,
            )
        except Exception:
            raise

        try:
            sdk.inference.virtual_models.retrieve(name=name, workspace=workspace)
            return
        except NotFoundError as e:
            last_error = e
            time.sleep(poll_interval)
            continue
        except Exception:
            raise

    raise TimeoutError(f"VirtualModel {workspace}/{name} not available after {timeout}s. Last error: {last_error}")


def short_unique_name(prefix: str, max_length: int = 32) -> str:
    """Generate a short unique name with a prefix and random suffix.

    Useful for creating unique workspace names, entity names, or other identifiers
    in tests that have length constraints. The result conforms to entity name
    rules: lowercase, starts with a letter, 2-63 chars, no consecutive hyphens,
    no trailing hyphen.

    Args:
        prefix: The prefix for the name (will be truncated and lowercased if needed)
        max_length: Maximum total length of the resulting name (default: 32)

    Returns:
        A unique name in the format "{prefix}-{random_suffix}"
    """
    suffix = uuid.uuid4().hex[:8]
    max_prefix_length = max_length - len(suffix) - 1
    base = prefix[:max_prefix_length].lower().rstrip("-")
    if not base or base[0].isdigit():
        base = "a" + (base[1:] if len(base) > 1 else "")
    base = re.sub(r"-+", "-", base).rstrip("-") or "a"
    return f"{base}-{suffix}"


def unique_email(prefix: str = "user") -> str:
    """Generate a unique email for test users.

    Args:
        prefix: The prefix for the email local part (default: "user")

    Returns:
        A unique email in the format "{prefix}-{random_suffix}@example.com"
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}@example.com"


def as_user(
    sdk: NeMoHelix,
    email: str,
    groups: list[str] | None = None,
) -> NeMoHelix:
    """Create a new SDK client authenticated as a specific user.

    The SDK uses immutable default_headers set at construction time.
    This function creates a new client with the specified principal headers.

    Args:
        sdk: The base SDK client to derive from
        email: The email/principal ID to authenticate as
        groups: Optional list of groups the user belongs to

    Returns:
        A new SDK client with auth headers set
    """
    headers: dict[str, str] = {
        "X-NHX-Principal-Id": email,
    }
    if "@" in email:
        headers["X-NHX-Principal-Email"] = email
    if groups:
        headers["X-NHX-Principal-Groups"] = ",".join(groups)
    return sdk.with_options(set_default_headers=headers)


def as_service_for(
    sdk: NeMoHelix,
    on_behalf_of: str,
    service_name: str = "platform",
) -> NeMoHelix:
    """Create a new SDK client authenticated as a service principal acting on behalf of a user.

    Mirrors the production pattern where services call internal APIs (e.g., the
    generic Entities API) as ``service:<name>`` with the real user identity in
    ``X-NHX-Principal-On-Behalf-Of``.  The service principal bypasses OPA
    permission checks while the on-behalf-of header preserves audit attribution.

    Args:
        sdk: The base SDK client to derive from
        on_behalf_of: The user's principal ID (typically email) to delegate for
        service_name: Service identity suffix (default: "platform")

    Returns:
        A new SDK client with service principal + on-behalf-of headers
    """
    return sdk.with_options(
        set_default_headers={
            "X-NHX-Principal-Id": f"service:{service_name}",
            "X-NHX-Principal-On-Behalf-Of": on_behalf_of,
        }
    )


def grant_workspace_role(
    sdk: NeMoHelix,
    *,
    workspace: str,
    principal: str,
    roles: Sequence[str],
    wait_role_propagation: bool = True,
) -> WorkspaceMember:
    """Grant workspace roles to a principal in auth-enabled tests."""
    return sdk.workspaces.members.create(
        workspace=workspace,
        principal=principal,
        roles=list(roles),
        wait_role_propagation=wait_role_propagation,
    )


@contextmanager
def igw_mock_provider_mode(prefix: str = "igw-mock-") -> Iterator[None]:
    """Enable IGW mock-provider mode in *this* process for the duration of the ``with`` block.

    Overrides ``InferenceGatewayConfig.mock_provider_prefix`` so :func:`add_mock_provider` can name
    and resolve mock providers. Use this when driving a *real* platform subprocess (where
    :func:`create_test_client` isn't in play) but the in-process config still has to recognize the
    mock prefix.

    Implemented as a ``Configuration`` override, which ``get_service_config`` honors ahead of the
    env-derived config — so it works regardless of whether the IGW config was already read/cached
    (an env var would lose that race) — and is scoped to the block, so it can't leak into unrelated
    suites. The prior ``InferenceGatewayConfig`` override (if any) is restored on exit.
    """
    from nhx.common.config import Configuration
    from nhx.core.inference_gateway.config import InferenceGatewayConfig

    current = Configuration.get_service_config(InferenceGatewayConfig)
    merged = InferenceGatewayConfig(**{**current.model_dump(), "mock_provider_prefix": prefix})
    Configuration.set_overrides({InferenceGatewayConfig: merged})
    try:
        yield
    finally:
        Configuration.clear_override(InferenceGatewayConfig)


def add_mock_provider(
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str | None = None,
    mock_response_body: dict[str, Any] | None = None,
    mock_response_body_by_model: dict[str, list[MockProviderResponse]] | None = None,
    mock_status: int | None = None,
    host_url: str = "http://mock.local",
    served_models: dict[str, str] | None = None,
    enabled_models: list[str] | None = None,
    should_autoprovision_virtual_model: bool = True,
) -> ModelProvider:
    """Add a mock provider via the SDK API.

    The provider name will be prefixed with the configured mock_provider_prefix
    (typically "igw-mock-") to ensure it's recognized as a mock provider by the IGW.

    When used in integration tests, this function requires that create_test_client was called
    with igw_mock_provider_mode=True, which sets the mock_provider_prefix on the InferenceGatewayConfig.

    When used in E2E tests, this function requires that the NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX
    environment variable is manually set to a mock_provider_prefix value.

    To return a mock static response, set the `mock_response_body` parameter.
    To return dynamic responses based on the model name, set the `mock_response_body_by_model` parameter.

    Args:
        sdk: The NeMoHelix SDK client.
        workspace: Provider workspace. Must be a valid entity name (see NAME_PATTERN).
        name: Provider name (will be auto-prefixed with "igw-mock-"). Must be a valid
            entity name so that the prefixed name matches NAME_PATTERN. If omitted, a
            unique name is generated automatically — recommended when tests share a
            workspace (ex. in Kubernetes e2e runs) to avoid 409 conflicts.
        mock_response_body: Optional mock response JSON body. If provided, all requests
            to this provider return this response. If None, the provider uses smart
            defaults (health, models) or requires X-Mock-Response header on each request.
        mock_response_body_by_model: Optional dict mapping model name to list of responses.
            If provided, responses are returned sequentially per model (clamping to last
            if exhausted). This is useful for testing Guardrails, where a single API call may invoke
            a single model multiple times. Format: `{"workspace/model": [response1, response2], ...}`.
            Each response must be a MockProviderResponse (raw JSON body and status code).
            Cannot be used together with `mock_response_body`.
        mock_status: Optional HTTP status code for the mock response (default: 200).
        host_url: Host URL (not used in mock mode, but required by schema).
        served_models: Optional dict mapping model_entity_name to served_model_name for
            model entity and OpenAI routing. Keys must be valid entity names (see
            NAME_PATTERN: start with [a-z], 2-63 chars, lowercase letters/digits/hyphens,
            no consecutive hyphens). If None, a default mapping is created using the
            `name` parameter (unprefixed) as both entity and served model name.
        enabled_models: Optional list of enabled models. If None, all models are enabled.
        should_autoprovision_virtual_model: Whether helper-created passthrough
            VirtualModels should be marked controller-managed. Keep the default True
            to match production provider reconciler behavior. Set False in tests
            that need these mock routes to survive the controller's orphan cleanup
            window.

    Returns:
        The created ModelProvider object (with prefixed name).

    Raises:
        RuntimeError: If mock_provider_prefix is not configured (igw_mock_provider_mode not enabled).
        ValueError: If both mock_response_body and mock_response_body_by_model are provided;
            if workspace or name (after prefix) is not a valid entity name; or if any
            served_models key is not a valid entity name (see NAME_PATTERN).

    Example (static response):
        with create_test_client(
            InferenceGatewayService, client_type=ClientContext, igw_mock_provider_mode=True
        ) as ctx:
            # Simple usage - default model entity mapping created automatically
            provider = add_mock_provider(
                ctx.sdk,
                workspace="default",
                name="judge",  # Becomes "igw-mock-judge"
                mock_response_body={
                    "id": "chatcmpl-mock",
                    "choices": [{"message": {"role": "assistant", "content": "..."}}],
                },
            )
            # Provider route
            response = ctx.sdk.inference.gateway.provider.post(
                "v1/chat/completions",
                name=provider.name,
                workspace="default",
                body={"model": "test", "messages": []},
            )
            # Model entity route (uses default mapping: entity="judge", served="judge")
            response = ctx.sdk.inference.gateway.model.post(
                "v1/chat/completions",
                name="judge",
                workspace="default",
                body={"model": "test", "messages": []},
            )
    """
    # Import IGW dependencies here to avoid circular imports at module load time
    from nhx.common.config import Configuration
    from nhx.core.inference_gateway.api.dependencies import global_model_cache, global_virtual_model_cache
    from nhx.core.inference_gateway.api.mock_provider import (
        MOCK_RESPONSE_HEADER,
        MOCK_RESPONSE_MAP_HEADER,
        MOCK_SERVED_MODELS_HEADER,
        MOCK_STATUS_HEADER,
    )
    from nhx.core.inference_gateway.api.model_cache import ModelProviderInfo
    from nhx.core.inference_gateway.config import InferenceGatewayConfig

    if not _ENTITY_NAME_PATTERN.match(workspace):
        raise ValueError(f"Invalid workspace {workspace!r}. {NAME_PATTERN_DESCRIPTION}")

    # Validate mutually exclusive parameters
    if mock_response_body is not None and mock_response_body_by_model is not None:
        raise ValueError(
            "Cannot specify both `mock_response_body` and `mock_response_body_by_model`. "
            "Use `mock_response_body` for static responses, or `mock_response_body_by_model` for dynamic responses."
        )

    # Get the mock provider prefix from configuration.
    # For integration tests: set by create_test_client(igw_mock_provider_mode=True).
    # For E2E tests: environment variable is explicitly set.
    igw_config = Configuration.get_service_config(InferenceGatewayConfig)
    prefix = igw_config.mock_provider_prefix
    if prefix is None:
        raise RuntimeError(
            "mock_provider_prefix is not configured. "
            "For integration tests, call create_test_client with igw_mock_provider_mode=True. "
            "For E2E tests, set the NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX "
            "environment variable."
        )

    # Generate a unique name if none was provided
    if name is None:
        name = short_unique_name("provider")

    # Auto-prefix the name if not already prefixed
    prefixed_name = name if name.startswith(prefix) else f"{prefix}{name}"

    if not _ENTITY_NAME_PATTERN.match(prefixed_name):
        raise ValueError(f"Invalid name {prefixed_name!r}. {NAME_PATTERN_DESCRIPTION}")

    # Compute served_models defaults before building default_extra_headers: the provider
    # reconciler calls GET /v1/models on each provider, and MOCK_SERVED_MODELS_HEADER tells
    # the mock which IDs to return. Without it, the mock returns the generic "mock-model"
    # default, causing the reconciler to overwrite our update_status served_models mapping.
    # Use model *entity* names (the served_models keys), not served_model_name values:
    # discovery builds model_entity_id from the mock's /v1/models ids, and passthrough
    # VirtualModels are named after that entity. Advertising served names here makes the
    # reconciler delete VMs created for the entity and recreate them under the wrong name.
    if served_models is None:
        if mock_response_body_by_model:
            served_models = {
                model_name.split("/")[-1]: model_name.split("/")[-1]
                for model_name in mock_response_body_by_model.keys()
            }
        else:
            served_models = {name: name}  # Use unprefixed name as both entity and served model

    # Build default_extra_headers based on the mock response type (if provided)
    default_extra_headers: dict[str, str] = {}
    if mock_response_body_by_model is not None:
        # Dynamic per-model responses
        default_extra_headers[MOCK_RESPONSE_MAP_HEADER] = _serialize_mock_response_map(mock_response_body_by_model)
    elif mock_response_body is not None:
        # Static response
        default_extra_headers[MOCK_RESPONSE_HEADER] = json.dumps(mock_response_body)
        if mock_status is not None:
            default_extra_headers[MOCK_STATUS_HEADER] = str(mock_status)

    default_extra_headers[MOCK_SERVED_MODELS_HEADER] = json.dumps(list(served_models.keys()))

    # Create the provider via SDK API (served_models not supported in SDK API).
    # If a provider with the same name already exists (ex. in a shared workspace),
    # delete it first so each test starts with a clean mock provider.

    try:
        provider = sdk.inference.providers.create(
            workspace=workspace,
            name=prefixed_name,
            host_url=host_url,
            default_extra_headers=default_extra_headers,
            enabled_models=enabled_models or omit,
        )
    except ConflictError:
        sdk.inference.providers.delete(workspace=workspace, name=prefixed_name)
        provider = sdk.inference.providers.create(
            workspace=workspace,
            name=prefixed_name,
            host_url=host_url,
            default_extra_headers=default_extra_headers,
            enabled_models=enabled_models or omit,
        )

    for entity_name in served_models:
        if not _ENTITY_NAME_PATTERN.match(entity_name):
            raise ValueError(
                f"served_models key {entity_name!r} is not a valid model entity name. {NAME_PATTERN_DESCRIPTION}"
            )

    served_model_mappings = [
        ServedModelMapping(
            model_entity_id=f"{workspace}/{entity_name}",
            served_model_name=served_name,
        )
        for entity_name, served_name in served_models.items()
    ]

    # Create provider with served_models for cache (SDK API doesn't return served_models)
    provider = ModelProvider(
        id=get_random_id("provider"),
        workspace=workspace,
        name=prefixed_name,
        host_url=host_url,
        default_extra_headers=default_extra_headers or None,
        served_models=served_model_mappings,
        enabled_models=enabled_models,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Always persist served_models via update_status so the Models API / IGW cache refresh
    # sees the correct mapping. Without this, refresh_model_cache overwrites the local cache
    # with providers from the API (which have empty served_models from create), causing 404
    # for model entity and OpenAI routes.
    sdk.inference.providers.update_status(
        name=prefixed_name,
        workspace=workspace,
        served_models=[
            {"model_entity_id": sm.model_entity_id, "served_model_name": sm.served_model_name}
            for sm in served_model_mappings
        ],
    )

    # Create a passthrough VirtualModel via the SDK for every served entity, mirroring
    # the production provider reconciler's _ensure_passthrough_virtual_model behavior.
    # The IGW now requires every inference request to resolve to a VirtualModel, and
    # the IGW's background cache refresher rebuilds the VM map from the SDK list every
    # few seconds — so any local-only seed would be wiped out at the next refresh tick.
    # Going through the SDK ensures the VM exists in the entity store and survives refreshes.
    # 409 ConflictError is treated as idempotent (matches the production reconciler).
    for entity_name in served_models:
        ensure_passthrough_virtual_model(
            sdk,
            workspace,
            entity_name,
            autoprovisioned=should_autoprovision_virtual_model,
        )

    try:
        # From integration tests, we can directly update the local model cache to speed up subsequent requests
        model_cache = global_model_cache()
        provider_info = ModelProviderInfo(model_provider=provider)
        model_cache.update_model_info(provider_info)
        model_cache.rebuild_model_entity_map()

        # Also seed the local VirtualModel cache so requests fired immediately after
        # this call hit the right cache state without waiting for the IGW's next
        # background refresh tick. This in-place seed is purely a latency optimization.
        from datetime import datetime as _datetime

        from nemo_helix.types.inference.virtual_model import VirtualModel as _SDKVirtualModel

        virtual_model_cache = global_virtual_model_cache()
        now = _datetime.now()
        for entity_name in served_models:
            key = (workspace, entity_name)
            if key in virtual_model_cache.virtual_model_map:
                continue
            virtual_model_cache.virtual_model_map[key] = _SDKVirtualModel(
                id=f"{workspace}/{entity_name}",
                entity_id=f"{workspace}/{entity_name}",
                workspace=workspace,
                name=entity_name,
                parent=workspace,
                db_version=1,
                default_model_entity=f"{workspace}/{entity_name}",
                autoprovisioned=should_autoprovision_virtual_model,
                created_at=now,
                updated_at=now,
            )
    except RuntimeError:
        # From E2E tests, the local cache is not available (app runs in a separate process).
        # Wait for model entities AND their autoprovisioned VirtualModels to become
        # available in the remote IGW caches. The provider reconciler creates passthrough
        # VirtualModels asynchronously after update_status; inference routes require both
        # the model entity and its VirtualModel to be visible before requests succeed.
        for entity_name in served_models.keys():
            wait_for_model_entity(
                sdk,
                workspace,
                entity_name,
                timeout=60,
                ensure_virtual_model=True,
                should_autoprovision_virtual_model=should_autoprovision_virtual_model,
            )
            ensure_passthrough_virtual_model(
                sdk,
                workspace,
                entity_name,
                timeout=60,
                autoprovisioned=should_autoprovision_virtual_model,
            )

    return provider
