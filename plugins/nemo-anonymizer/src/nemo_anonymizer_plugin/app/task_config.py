# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing and internal config types for anonymizer requests."""

from __future__ import annotations

from typing import Any

import data_designer.config as dd
from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, ReplaceMethodBase, Substitute
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec
from nemo_anonymizer_plugin.app.model_configs import SelectedModelsOverrides
from nemo_anonymizer_plugin.config import DEFAULT_MAX_PREVIEW_NUM_RECORDS
from pydantic import BaseModel, Field, field_serializer

_REPLACE_METHOD_KINDS: dict[type[ReplaceMethodBase], str] = {
    Annotate: "annotate",
    Hash: "hash",
    Redact: "redact",
    Substitute: "substitute",
}


class AnonymizerRequest(BaseModel):
    """User-facing anonymizer execution request.

    Fields:
      config:           AnonymizerConfig — replace/rewrite mode + detection params.
      data:             AnonymizerInputSpec — HTTP(S) URL or fileset source + text/id columns.
      model_configs:    DD ``ModelConfig`` list. ``provider`` on each entry must
                        reference a NeMo Helix inference provider name (optionally
                        ``workspace/provider``). Optional on the request model for
                        compatibility, but required by plugin preview/run execution
                        so requests route through NeMo Helix Inference Gateway.
      selected_models:  Optional role->alias overrides. Omitted roles fall back
                        to the upstream library YAML defaults.
    """

    model_config = {"json_schema_mode_override": "validation"}

    config: AnonymizerConfig
    data: AnonymizerInputSpec
    model_configs: list[dd.ModelConfig] | None = None
    selected_models: SelectedModelsOverrides | None = None

    @field_serializer("config")
    def serialize_config(self, config: AnonymizerConfig) -> dict:
        payload = config.model_dump(mode="json", exclude_none=True)
        if config.replace is not None:
            payload["replace"] = _serialize_replace_method(config.replace)
        return payload


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
    # Provider definitions resolved against NeMo Helix. Each entry already points at
    # the Inference Gateway URL with the right auth headers. The task will pass
    # these to ``Anonymizer(model_providers=...)``.
    dd_model_providers: list[dict[str, Any]]


def _serialize_replace_method(replace: ReplaceMethodBase) -> dict:
    payload = replace.model_dump(mode="json", exclude_none=True)
    for replace_type, kind in _REPLACE_METHOD_KINDS.items():
        if isinstance(replace, replace_type):
            return {"kind": kind, **payload}
    return payload
