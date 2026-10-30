# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models used by inference middleware configuration and VirtualModels."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any, Self

from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.inference_middleware import BackendFormat
from pydantic import BaseModel, Field, ModelWrapValidatorHandler, TypeAdapter, model_validator

GUARDRAIL_CONFIG_TYPE = "guardrail_config"
"""``MiddlewareCall.config_type`` discriminator for a stored NeMo Guardrails config.

Mirrors ``GuardrailConfig.__entity_type__`` in the Guardrails service and
``GUARDRAILS_PLUGIN_CONFIG_TYPE`` in the guardrails middleware plugin.  Declared here so
:class:`VirtualModel` can derive :attr:`VirtualModel.guardrail_config_ids` without importing
either of them.
"""


class MiddlewareCall(BaseModel):
    """One entry in a VirtualModel middleware pipeline.

    Declares which plugin to invoke and how to resolve its configuration.
    Exactly one of ``config`` (inline dict) or ``config_id`` (entity reference)
    should be provided. ``config_type`` is always required regardless of which
    is used — it is the discriminator that tells IGW (and the plugin) which
    config schema applies.

    Attributes:
        name: The entry-point key of the plugin to invoke
            (e.g. ``"nemo-switchyard"``). Must match the plugin's
            ``nemo.inference_middleware`` entry-point key.
        config_type: Always required. Maps to the ``entity_type`` of the plugin's
            config ``NemoEntity`` subclass (e.g. ``"routellm_config"``). Used by
            IGW to call :meth:`~NemoInferenceMiddleware.validate_middleware_config`
            with the right discriminator, and by the plugin to dispatch to the
            correct schema when it supports multiple config types.
        config: Inline config dict. Mutually exclusive with ``config_id``.
        config_id: ``"workspace/name"`` reference to a stored config entity.
            Mutually exclusive with ``config``. IGW resolves this by calling
            :meth:`~NemoInferenceMiddleware.get_middleware_config` on the plugin.
    """

    name: str
    config_type: str
    config: dict[str, Any] | None = None
    config_id: str | None = None

    @model_validator(mode="after")
    def _ensure_exactly_one_config_source(self) -> Self:
        if (self.config is None) == (self.config_id is None):
            raise ValueError("Exactly one of config or config_id must be provided")
        return self


class VirtualModelInferenceConfig(BaseModel):
    """Inference configuration for one model entity referenced by a VirtualModel."""

    model: str
    """Model entity reference in ``"workspace/name"`` format."""

    backend_format: BackendFormat | None = Field(
        default=None,
        description="Optional backend format override for this VirtualModel entry.",
        json_schema_extra={"nullable": True},
    )


_AUTOPROVISIONED_DESC = (
    "Marks this VirtualModel as controller-managed. The Models controller will delete it once no "
    "ModelProvider serves the matching entity. Setting this manually opts the VirtualModel into "
    "that cleanup behavior."
)


class VirtualModel(NemoEntity, entity_type="virtual_model"):
    """Logical inference route.

    Maps a user-facing model name to an optional default model entity and
    defines ordered middleware pipelines for the request, response, and
    post-response phases.

    When a caller sets ``model: "workspace/my-virtual-model"`` in an inference
    request, IGW resolves the ``VirtualModel`` instead of a ``ModelEntity``
    directly. If ``default_model_entity`` is set, IGW writes it into
    ``request["model"]`` before the request middleware pipeline runs. Middleware
    may mutate ``request["model"]`` freely. After the pipeline completes, IGW
    reads ``request["model"]``, resolves it to a ``ModelProvider`` via the
    ``ModelCache``, and proxies.

    The ``ModelProviderReconciler`` auto-creates a passthrough ``VirtualModel``
    for each discovered model (same workspace and name as the ``ModelEntity``,
    empty middleware lists, ``default_model_entity`` pointing to that entity).
    All existing inference requests continue to work without changes.
    """

    default_model_entity: str | None = None
    """``"workspace/model-entity-name"`` written into ``request["model"]`` before
    the request middleware pipeline runs. If ``None``, no value is written — a
    request middleware plugin must handle the backend call itself and return an
    :class:`InferenceResponse` or ``AsyncIterator``."""

    autoprovisioned: bool = Field(
        default=False,
        description=_AUTOPROVISIONED_DESC,
    )
    """Whether this VirtualModel was automatically created by the
    ModelProviderReconciler for a discovered model entity."""

    models: list[VirtualModelInferenceConfig] = Field(default_factory=list)
    """Model entity references used by this VirtualModel. A per-entry
    ``backend_format`` overrides the referenced ModelEntity value for requests
    resolved through this VirtualModel."""

    request_middleware: list[MiddlewareCall] = []
    """Ordered list of middleware plugins applied before proxying."""

    response_middleware: list[MiddlewareCall] = []
    """Ordered list of middleware plugins applied after the backend response is
    received, before returning it to the caller."""

    post_response_middleware: list[MiddlewareCall] = []
    """Ordered list of middleware plugins invoked after the response has been
    returned to the caller. Intended for fire-and-forget work (e.g. logging,
    analytics) that must not block or modify the response."""

    override_proxy: str | None = None
    """Optional. Names a plugin-provided proxy implementation IGW should use
    instead of its default ``aiohttp`` proxy. Format: ``"plugin-name.proxy-name"``.
    If unset, IGW performs the proxy itself."""

    guardrail_config_ids: list[str] = Field(
        default_factory=list,
        description=(
            "System-managed. Guardrail configs applied by this VirtualModel's middleware, as "
            '"workspace/name" references. Derived from the middleware pipelines on every write and '
            "ignored if supplied on a create or update body. Filter on it with filter[guardrail_config]."
        ),
    )
    """System-managed. Distinct ``"workspace/name"`` guardrail config references reached by
    this VirtualModel's middleware pipelines, in first-seen order."""

    def middleware_calls(self) -> Iterator[MiddlewareCall]:
        """Every middleware call across the request, response, and post-response pipelines."""
        yield from self.request_middleware or []
        yield from self.response_middleware or []
        yield from self.post_response_middleware or []

    def references_guardrail_config(self, config_id: str) -> bool:
        """Whether any middleware call resolves the stored guardrail config ``config_id``."""
        return any(
            call.config_type == GUARDRAIL_CONFIG_TYPE and call.config_id == config_id
            for call in self.middleware_calls()
        )

    def refresh_guardrail_config_ids(self) -> Self:
        """Recompute :attr:`guardrail_config_ids` from the middleware pipelines, in place."""
        seen: dict[str, None] = {}
        for call in self.middleware_calls():
            if call.config_type == GUARDRAIL_CONFIG_TYPE and call.config_id:
                seen[call.config_id] = None
        self.guardrail_config_ids = list(seen)
        return self

    @model_validator(mode="after")
    def _sync_guardrail_config_ids(self) -> Self:
        """Keep the denormalized reference list in step with the pipelines it is derived from."""
        return self.refresh_guardrail_config_ids()

    @model_validator(mode="wrap")
    @classmethod
    def _hydrate_wire_metadata(
        cls,
        value: object,
        handler: ModelWrapValidatorHandler[Self],
    ) -> Self:
        """Retain entity metadata when validating a VirtualModel API response."""
        model = handler(value)
        if not isinstance(value, dict):
            return model

        wire_data = TypeAdapter(dict[str, object]).validate_python(value)
        optional_string = TypeAdapter(str | None)
        optional_datetime = TypeAdapter(datetime | None)
        if "id" in wire_data:
            model._id = optional_string.validate_python(wire_data["id"])
        if "created_at" in wire_data:
            model._created_at = optional_datetime.validate_python(wire_data["created_at"])
        if "updated_at" in wire_data:
            model._updated_at = optional_datetime.validate_python(wire_data["updated_at"])
        if "created_by" in wire_data:
            model._created_by = optional_string.validate_python(wire_data["created_by"])
        if "updated_by" in wire_data:
            model._updated_by = optional_string.validate_python(wire_data["updated_by"])
        if "parent" in wire_data:
            model._parent = optional_string.validate_python(wire_data["parent"])
        return model


GUARDRAIL_CONFIG_IDS_FIELD = "data.guardrail_config_ids"
"""Entity-store path of :attr:`VirtualModel.guardrail_config_ids`."""

_LEGACY_MIDDLEWARE_FIELDS = (
    "data.request_middleware",
    "data.response_middleware",
    "data.post_response_middleware",
)
"""Pipelines that ``guardrail_config_ids`` is derived from, scanned directly for rows written
before that field existed."""


def guardrail_config_membership_filter(config_id: str) -> LogicalOperation:
    """Entity-store predicate for "this VirtualModel applies guardrail config ``config_id``"."""
    return LogicalOperation(
        operator=FilterOperator.OR,
        operations=[
            ComparisonOperation(operator=FilterOperator.CONTAINS, field=GUARDRAIL_CONFIG_IDS_FIELD, value=config_id),
            LogicalOperation(
                operator=FilterOperator.AND,
                operations=[
                    ComparisonOperation(operator=FilterOperator.EQ, field=GUARDRAIL_CONFIG_IDS_FIELD, value=None),
                    LogicalOperation(
                        operator=FilterOperator.OR,
                        operations=[
                            ComparisonOperation(operator=FilterOperator.CONTAINS, field=field, value=config_id)
                            for field in _LEGACY_MIDDLEWARE_FIELDS
                        ],
                    ),
                ],
            ),
        ],
    )
