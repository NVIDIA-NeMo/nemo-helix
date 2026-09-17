# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared producer pass for runnable Data Designer configs.

Three call sites in the plugin run the same first three checks against a
``DataDesignerConfig`` before doing their own thing:

- ``PreviewFunction.run`` (runtime — generates the preview).
- ``CreateJob.to_spec`` (runtime — compiles the job step config).
- ``validate_config`` (reporter — emits a ``ValidationReport``).

The runtime call sites need ``(model_configs, model_providers)`` to drive
their next step; the reporter only wants the error list. This module owns
the shared pass and returns the error buffer plus the resolved values, so
runtime callers can ``raise_if_errors`` and use the values, while the
reporter can project the errors into a structured result and additionally
run engine-level checks of its own.
"""

from __future__ import annotations

import data_designer.config as dd
from data_designer_nemo.context import DataDesignerValidationContext
from data_designer_nemo.errors import NDDError, NDDInternalError, NDDInvalidConfigError
from data_designer_nemo.model_configs import get_model_configs


async def resolve_runnable_config(
    dd_ctx: DataDesignerValidationContext,
    config: dd.DataDesignerConfig,
) -> tuple[list[NDDError], list[dd.ModelConfig], list[dd.ModelProvider]]:
    """Run the producer pass against ``config`` and return ``(errors, model_configs, model_providers)``.

    Steps, in order:

    1. ``dd_ctx.validate(config)`` — runs every sub-check the context owns
       (seed type, tool configs, IGW personas filesets, etc.) and accumulates
       the errors.
    2. ``get_model_configs(config)`` — extracts every ``ModelConfig``
       referenced by columns / profilers. ``NDDInvalidConfigError`` (e.g.
       unrecognized aliases) is appended; we then **skip** step 3 because
       provider resolution requires resolved aliases.
    3. ``dd_ctx.get_model_providers(model_configs)`` — resolves provider
       references against locally-defined providers and the Inference
       Gateway. Appends ``NDDInvalidConfigError`` /
       ``NDDInternalError`` on failure.

    Never raises ``NDDError`` — every problem ends up in the returned buffer
    instead. Truly unexpected errors (programmer bugs, transport-layer
    failures outside the validators themselves) propagate as exceptions.

    Callers decide what to do with the result:

    - Runtime callers (``CreateJob.to_spec`` / ``PreviewFunction.run``) call
      :func:`raise_if_errors` on the buffer and then use ``model_configs``
      / ``model_providers`` to drive their next step.
    - The reporter (``validate_config`` in the plugin) maps the buffer into
      its public ``ValidationError`` shape and additionally runs an
      engine-level compile check when the buffer is empty.
    """
    errors: list[NDDError] = list(await dd_ctx.validate(config))

    model_configs: list[dd.ModelConfig] = []
    model_providers: list[dd.ModelProvider] = []

    try:
        model_configs = get_model_configs(config)
    except NDDInvalidConfigError as e:
        errors.append(e)
    else:
        try:
            model_providers = await dd_ctx.get_model_providers(model_configs)
        except (NDDInvalidConfigError, NDDInternalError) as e:
            errors.append(e)

    return errors, model_configs, model_providers
