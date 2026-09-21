# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing and internal config types for anonymizer requests."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union, cast

import data_designer.config as dd
from anonymizer.config.anonymizer_config import AnonymizerConfig, Detect, Rewrite
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, ReplaceMethod, ReplaceMethodBase, Substitute
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec
from nemo_anonymizer_plugin.app.model_configs import SelectedModelsOverrides
from nemo_anonymizer_plugin.config import DEFAULT_MAX_PREVIEW_NUM_RECORDS
from pydantic import BaseModel, Field, Tag, model_validator

# The upstream `ReplaceMethod` union resolves `kind` through a callable
# `Discriminator` that reads a dict key, so `kind` never became a modelled
# field and is invisible to OpenAPI / generated SDKs (ASTD-329). A dict
# missing it also makes that callable raise a bare `TypeError`, which
# propagates as an unhandled 500 instead of a validation error (ASTD-328).
# These wire-level subclasses restate `kind` as a real `Literal` field so
# pydantic's own tagged-union discriminator — not a callable — resolves and
# validates it, giving both problems the same fix.


class SubstituteRequest(Substitute):
    """Replace entities with LLM-generated synthetic values."""

    kind: Literal["substitute"] = "substitute"


class RedactRequest(Redact):
    """Replace each entity with a configurable redaction template."""

    kind: Literal["redact"] = "redact"


class AnnotateRequest(Annotate):
    """Tag each entity with a readable label token."""

    kind: Literal["annotate"] = "annotate"


class HashRequest(Hash):
    """Replace each entity with a deterministic hash token."""

    kind: Literal["hash"] = "hash"


_REPLACE_METHOD_KINDS: dict[type[ReplaceMethodBase], str] = {
    Annotate: "annotate",
    Hash: "hash",
    Redact: "redact",
    Substitute: "substitute",
}

_REPLACE_UPSTREAM_TYPES: dict[type, type[ReplaceMethodBase]] = {
    SubstituteRequest: Substitute,
    RedactRequest: Redact,
    AnnotateRequest: Annotate,
    HashRequest: Hash,
}

ReplaceMethodRequest = Annotated[
    Union[
        Annotated[SubstituteRequest, Tag("substitute")],
        Annotated[RedactRequest, Tag("redact")],
        Annotated[AnnotateRequest, Tag("annotate")],
        Annotated[HashRequest, Tag("hash")],
    ],
    Field(discriminator="kind"),
]


def _wire_replace_payload(replace: ReplaceMethodBase) -> dict:
    """Convert an upstream ``ReplaceMethodBase`` instance into a `kind`-tagged dict."""
    payload = replace.model_dump(mode="python", exclude_none=True)
    for replace_type, kind in _REPLACE_METHOD_KINDS.items():
        if isinstance(replace, replace_type):
            return {"kind": kind, **payload}
    return payload


def _to_upstream_replace(replace: ReplaceMethodRequest) -> ReplaceMethod:
    upstream_type = _REPLACE_UPSTREAM_TYPES[type(replace)]
    return cast(ReplaceMethod, upstream_type(**replace.model_dump(exclude={"kind"})))


class AnonymizerConfigRequest(BaseModel):
    """Wire-level mirror of ``AnonymizerConfig``.

    Restates ``replace`` as a proper ``kind``-discriminated union so the
    OpenAPI schema and generated SDK types are self-describing (ASTD-329),
    and so a missing/invalid ``kind`` fails pydantic validation (422)
    instead of the upstream callable discriminator's bare ``TypeError``
    (ASTD-328). Convert to the upstream type with ``to_anonymizer_config()``
    before handing it to the ``anonymizer`` library.
    """

    detect: Detect = Field(default_factory=Detect, description="Entity detection configuration.")
    replace: ReplaceMethodRequest | None = Field(
        default=None,
        description="Replacement method (Substitute(), Redact(), Annotate(), or Hash()).",
    )
    rewrite: Rewrite | None = Field(default=None, description="Optional rewrite-mode parameters. ")
    emit_telemetry: bool = Field(
        default=True,
        description=(
            "Whether to emit anonymous Anonymizer telemetry events. See the Telemetry section "
            "in the README for what is collected and how to opt out at the environment or CLI level."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_upstream_config(cls, data: Any) -> Any:
        if isinstance(data, AnonymizerConfig):
            payload = data.model_dump(mode="python", exclude_none=True, exclude={"replace"})
            if data.replace is not None:
                payload["replace"] = _wire_replace_payload(data.replace)
            return payload
        return data

    def to_anonymizer_config(self) -> AnonymizerConfig:
        return AnonymizerConfig(
            detect=self.detect,
            replace=_to_upstream_replace(self.replace) if self.replace is not None else None,
            rewrite=self.rewrite,
            emit_telemetry=self.emit_telemetry,
        )


class AnonymizerRequest(BaseModel):
    """User-facing anonymizer execution request.

    Fields:
      config:           AnonymizerConfigRequest — replace/rewrite mode + detection params.
      data:             AnonymizerInputSpec — HTTP(S) URL or fileset source + text/id columns.
      model_configs:    DD ``ModelConfig`` list. ``provider`` on each entry must
                        reference a NeMo Platform inference provider name (optionally
                        ``workspace/provider``). Optional on the request model for
                        compatibility, but required by plugin preview/run execution
                        so requests route through NeMo Platform Inference Gateway.
      selected_models:  Optional role->alias overrides. Omitted roles fall back
                        to the upstream library YAML defaults.
    """

    model_config = {"json_schema_mode_override": "validation"}

    config: AnonymizerConfigRequest
    data: AnonymizerInputSpec
    model_configs: list[dd.ModelConfig] | None = None
    selected_models: SelectedModelsOverrides | None = None


class PreviewRequest(AnonymizerRequest):
    # Annotated, not `le=`: deployments may raise `preview_num_records.max` past this.
    num_records: int = Field(
        default=DEFAULT_MAX_PREVIEW_NUM_RECORDS,
        ge=1,
        json_schema_extra={"maximum": DEFAULT_MAX_PREVIEW_NUM_RECORDS},
    )


class AnonymizerStepConfig(BaseModel):
    """Internal carrier passed to the task container for ``anonymizer.run``."""

    model_config = {"json_schema_mode_override": "validation"}

    request: AnonymizerRequest
    # YAML body to hand to ``Anonymizer(model_configs=...)`` after the service
    # resolved providers and roles.
    model_configs_yaml: str
    # Provider definitions resolved against NeMo Platform. Each entry already points at
    # the Inference Gateway URL with the right auth headers. The task will pass
    # these to ``Anonymizer(model_providers=...)``.
    dd_model_providers: list[dict[str, Any]]
