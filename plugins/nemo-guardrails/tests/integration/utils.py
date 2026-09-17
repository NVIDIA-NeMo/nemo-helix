# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common functions shared by the nemo-guardrails plugin integration tests.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from typing import Any, TypedDict

import httpx
from nemo_guardrails_plugin.constants import GUARDRAILS_PLUGIN_CONFIG_TYPE
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.guardrail.client import GuardrailClient
from nemo_platform_plugin.guardrail.types import (
    CreateGuardrailConfigRequest,
    GuardrailConfig,
    UpdateGuardrailConfigRequest,
)
from nemo_platform_plugin.virtual_models.client import VirtualModelsClient
from nemo_platform_plugin.virtual_models.types import UpdateVirtualModelRequest
from nmp.testing.utils import short_unique_name
from typing_extensions import Required

DEFAULT_WORKSPACE = "default"
"""Default workspace seeded by the module-scoped IGW fixture. Helpers
default to this so parametrise-time builders that don't have a harness
yet keep working; production test bodies pass ``harness.workspace``
explicitly so the helpers stay portable if the fixture ever moves to
per-test workspaces."""

GUARDRAILS_PLUGIN_NAME = "nemo-guardrails"


@dataclass(frozen=True)
class ServedModel:
    served_name: str
    entity_ref: str


@dataclass(frozen=True)
class GuardrailsTestDataNames:
    test_id: str
    main_model_served_name: str
    main_model_entity_ref: str
    request_virtual_model_name: str
    guardrail_config_name: str
    model_provider_name: str


@dataclass(frozen=True)
class HarnessHTTPError:
    status_code: int
    body: object | None


class GuardrailsMiddlewareCall(TypedDict, total=False):
    name: Required[str]
    config_type: Required[str]
    config: dict[str, object]
    config_id: str


EntityGuardrailsMiddlewareCall = GuardrailsMiddlewareCall
InlineGuardrailsMiddlewareCall = GuardrailsMiddlewareCall


def expect_harness_http_error(action: Callable[[], object], expected_status: HTTPStatus) -> HarnessHTTPError:
    """Run a harness action and return HTTP error details.

    The IGW integration harness raises generated SDK HTTP errors for sync SDK
    calls and ``httpx.HTTPStatusError`` for direct TestClient streaming calls.
    Keep that dynamic boundary here so tests assert typed status/body values
    without importing generated SDK errors in the plugin suite.
    """
    try:
        action()
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
        except ValueError:
            body = exc.response.text
        error = HarnessHTTPError(status_code=exc.response.status_code, body=body)
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            raise AssertionError(f"Expected HTTP error with status_code, got {type(exc).__name__}") from exc
        error = HarnessHTTPError(status_code=status_code, body=getattr(exc, "body", None))
    else:
        raise AssertionError(f"Expected HTTP {expected_status.value}, but request succeeded")

    assert error.status_code == expected_status
    return error


def make_served_model(
    *,
    test_id: str,
    prefix: str,
    workspace: str = DEFAULT_WORKSPACE,
) -> ServedModel:
    served_name = f"{prefix}-{test_id}"
    return ServedModel(served_name=served_name, entity_ref=f"{workspace}/{served_name}")


def make_guardrails_test_data_names(
    *,
    main_model_prefix: str = "main-model",
    workspace: str = DEFAULT_WORKSPACE,
) -> GuardrailsTestDataNames:
    """Build a unique set of names + entity refs for one test.

    Pass ``workspace=harness.workspace`` so ``main_model_entity_ref``
    lines up with the harness's workspace. The default is a safety
    net for parametrise-time callers without a harness in scope.
    """
    test_id = short_unique_name("test")
    main_model = make_served_model(test_id=test_id, prefix=main_model_prefix, workspace=workspace)
    return GuardrailsTestDataNames(
        test_id=test_id,
        main_model_served_name=main_model.served_name,
        main_model_entity_ref=main_model.entity_ref,
        request_virtual_model_name=f"gr-vm-{test_id}",
        guardrail_config_name=f"gr-config-{test_id}",
        model_provider_name=f"gr-provider-{test_id}",
    )


def detach_guardrail_config(harness: Any, config_name: str) -> None:
    """Remove every middleware call referencing ``config_name`` from the harness's VirtualModels.

    The Guardrails service refuses to delete a config that a VirtualModel still applies, so a
    test that deletes its config in ``finally`` has to detach first: harness cleanup deletes the
    VirtualModels it created, but that runs *after* the test's own teardown.

    Detaches rather than deletes so the harness's own bookkeeping stays intact — it still deletes
    the VirtualModels itself, and does not log a spurious "failed to delete" warning for a route
    a test removed behind its back.
    """
    config_ref = f"{harness.workspace}/{config_name}"
    phases = ("request_middleware", "response_middleware", "post_response_middleware")
    virtual_models_client = client_from_platform(harness.sdk, VirtualModelsClient)

    for workspace, name in harness.virtual_models:
        try:
            virtual_model = virtual_models_client.get_virtual_model(name=name, workspace=workspace).data()
        except NotFoundError:
            continue

        remaining: dict[str, list[Any]] = {}
        for phase in phases:
            calls = getattr(virtual_model, phase, None) or []
            kept = [
                call
                for call in calls
                if not (
                    getattr(call, "config_type", None) == GUARDRAILS_PLUGIN_CONFIG_TYPE
                    and getattr(call, "config_id", None) == config_ref
                )
            ]
            if len(kept) != len(calls):
                remaining[phase] = [call.model_dump(exclude_none=True) for call in kept]

        if not remaining:
            continue

        virtual_models_client.update_virtual_model(
            name=name,
            workspace=workspace,
            body=UpdateVirtualModelRequest.model_validate(remaining),
        ).data()


def guardrail_client(harness: Any) -> GuardrailClient:
    return client_from_platform(harness.sdk, GuardrailClient)


def create_guardrail_config(
    harness: Any,
    *,
    name: str,
    description: str | None,
    data: dict[str, Any],
) -> GuardrailConfig:
    return (
        guardrail_client(harness)
        .create_guardrail_config(
            workspace=harness.workspace,
            body=CreateGuardrailConfigRequest(name=name, description=description, data=data),
        )
        .data()
    )


def update_guardrail_config(
    harness: Any,
    *,
    name: str,
    description: str | None = None,
    data: dict[str, Any] | None = None,
) -> GuardrailConfig:
    return (
        guardrail_client(harness)
        .update_guardrail_config(
            workspace=harness.workspace,
            name=name,
            body=UpdateGuardrailConfigRequest(description=description, data=data),
        )
        .data()
    )


def delete_guardrail_config_if_present(harness: Any, config_name: str) -> None:
    detach_guardrail_config(harness, config_name)
    try:
        guardrail_client(harness).delete_guardrail_config(name=config_name, workspace=harness.workspace).data()
    except NotFoundError:
        pass


class RailType(str, Enum):
    """The rail directions a guardrail config can declare.

    Used by per-class ``_config_data`` builders to select which rail
    flows and prompts are wired into the generated config.
    """

    INPUT = "input"
    OUTPUT = "output"


def make_guardrail_config(
    workspace: str,
    name: str,
    *,
    data: dict[str, Any],
) -> GuardrailConfig:
    """Wrap a ``data`` block (the inner PlatformRailsConfig shape) in a GuardrailConfig envelope.

    The envelope (``id`` / ``entity_id`` / ``parent`` / timestamps) is
    irrelevant to plugin behaviour but required by the SDK type.
    """
    return GuardrailConfig.model_validate(
        {
            "id": f"{workspace}/{name}",
            "entity_id": f"{workspace}/{name}",
            "parent": workspace,
            "workspace": workspace,
            "name": name,
            "db_version": 1,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "data": data,
        }
    )


def make_middleware_call(config: GuardrailConfig) -> InlineGuardrailsMiddlewareCall:
    # validate_middleware_config expects the inner PlatformRailsConfig data
    # block (models / rails / prompts), not the entity envelope. The full
    # GuardrailConfig dump silently absorbs envelope fields into model_extra
    # and rails.rails ends up None, so the plugin's input rail gets bypassed.
    payload: dict[str, object] = config.data.model_dump(mode="json", exclude_none=True)

    # Thread the config name through as the inline diagnostic label so
    # log lines and the response's guardrails_data carry ``<inline:<name>>``
    # instead of the bare ``<inline>`` placeholder.
    if config.name:
        payload["name"] = config.name

    return {
        "name": GUARDRAILS_PLUGIN_NAME,
        "config_type": GUARDRAILS_PLUGIN_CONFIG_TYPE,
        "config": payload,
    }
