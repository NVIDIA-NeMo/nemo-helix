# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and validate the ``model_configs`` YAML for the Anonymizer library.

``Anonymizer(model_configs=...)`` takes a unified YAML (string or file path)
defining the model aliases and optional ``selected_models`` overrides. The
standalone library can use bundled defaults when this is omitted, but plugin
preview/run execution requires explicit entries so providers resolve through
NeMo Helix Inference Gateway.

The plugin accepts overrides as a loose dict (``SelectedModelsOverrides``)
because some roles take a scalar and others take a pool, then renders the
YAML and runs it through upstream's ``parse_model_configs`` so schema errors
surface at the plugin API boundary as ``AnonymizerInvalidConfigError``
instead of from inside the engine at runtime.

Defaults: when a role is omitted, upstream's
``anonymizer/engine/ndd/model_loader.py::load_default_model_selection`` reads
``default_model_configs/{detection,replace,rewrite}.yaml`` and
``_merge_selections`` layers user overrides on top.
"""

from __future__ import annotations

import logging
from typing import Any

import data_designer.config as dd
import yaml
from anonymizer.engine.ndd.model_loader import parse_model_configs
from nemo_anonymizer_plugin.app.errors import AnonymizerInvalidConfigError
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class SelectedModelsOverrides(BaseModel):
    """Partial role -> alias overrides for the three workflows.

    Each section is optional and is merged onto the bundled YAML defaults at
    parse time by upstream's ``anonymizer/engine/ndd/model_loader.py::_merge_selections``.
    Validation of the merged result still happens upstream and is also run
    early at the plugin boundary by ``build_model_configs_yaml``.
    """

    detection: dict[str, str | list[str]] | None = Field(default=None)
    replace: dict[str, str] | None = Field(default=None)
    rewrite: dict[str, str] | None = Field(default=None)


def build_model_configs_yaml(
    *,
    model_configs: list[dd.ModelConfig],
    selected_models: SelectedModelsOverrides | None = None,
) -> str:
    """Render and validate the unified YAML body for the Anonymizer library.

    The ``model_configs`` are emitted in the shape DD/Anonymizer's YAML
    loaders expect (alias/model/provider/inference_parameters). We dump the
    Pydantic models with ``mode="json"`` so enums and other custom types
    serialize cleanly.

    After rendering, the body is parsed via upstream's ``parse_model_configs``
    so that schema errors (unknown role names, wrong value types, malformed
    ``ModelConfig`` entries) surface at the plugin's API boundary as
    ``AnonymizerInvalidConfigError`` instead of bubbling up from inside the
    engine at runtime.
    """
    payload: dict[str, Any] = {
        "model_configs": [mc.model_dump(mode="json", exclude_none=True) for mc in model_configs],
    }
    if selected_models is not None:
        sm: dict[str, Any] = {}
        if selected_models.detection:
            sm["detection"] = dict(selected_models.detection)
        if selected_models.replace:
            sm["replace"] = dict(selected_models.replace)
        if selected_models.rewrite:
            sm["rewrite"] = dict(selected_models.rewrite)
        if sm:
            payload["selected_models"] = sm
    yaml_body = yaml.safe_dump(payload, sort_keys=False)

    try:
        parse_model_configs(yaml_body)
    except (ValidationError, ValueError, TypeError) as exc:
        raise AnonymizerInvalidConfigError(f"Invalid model_configs/selected_models payload: {exc}") from exc

    logger.info("Anonymizer model_configs YAML built successfully")
    return yaml_body


def has_selected_model_overrides(selected_models: SelectedModelsOverrides | None) -> bool:
    """Return true when the request contains at least one selected-model override."""
    return bool(selected_models and selected_models.model_dump(exclude_none=True))


def validate_selected_models_have_model_configs(
    *,
    model_configs: list[dd.ModelConfig] | None,
    selected_models: SelectedModelsOverrides | None,
) -> None:
    """Reject selected-model overrides without an explicit model pool.

    Upstream treats ``selected_models`` as a section in the same unified YAML
    document as ``model_configs``. If the plugin accepts overrides without a
    model pool, the only choices are to silently ignore them or synthesize an
    implicit default pool that bypasses NeMo Helix provider resolution. Failing
    fast keeps the user's intent explicit.
    """
    if has_selected_model_overrides(selected_models) and not model_configs:
        raise AnonymizerInvalidConfigError("selected_models requires model_configs so aliases can be resolved.")
