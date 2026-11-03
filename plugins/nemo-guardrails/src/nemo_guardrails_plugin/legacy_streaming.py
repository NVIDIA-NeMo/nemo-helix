# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensures configs with an output rail that rewrites the response load under nemoguardrails 0.24+.

Every config stored by the guardrails service has the default values of ``OutputRailsStreamingConfig``
written into ``rails.output.streaming``: ``enabled=True``, ``stream_first=True``, ``context_size=50``.
Entities are saved with their field defaults, so this holds even when the user never configured
streaming.

nemoguardrails 0.24 raises ``InvalidRailsConfigurationError`` for ``enabled=True`` together with
``stream_first=True`` or a non-zero ``context_size`` when any output flow rewrites the response (PII
masking, injection detection, etc.), because text that was already streamed can't be rewritten.

Fix: for a config with ``enabled=True`` and such a flow, set ``stream_first=False`` and change
``context_size`` from 50 to 0, which are the values the library requires. Any other ``context_size``
is kept, so the library still reports it.

Stored configs reach the plugin already repaired by the service's model. This copy covers configs
written inline on a VirtualModel, which never pass through that model.
"""

import logging
from typing import Any

from nemoguardrails.manifests import RailDirection, default_rail_catalog, parse_configured_surface

logger = logging.getLogger(__name__)

_LEGACY_CONTEXT_SIZE = 50
"""The ``context_size`` default stored on every existing config."""


def _output_flows_that_rewrite_response(flows: list[str]) -> list[str]:
    """The configured output flows whose rail rewrites the response, using the library's own catalog."""
    surfaces = default_rail_catalog().surfaces()
    rewriting = []
    for flow in flows:
        try:
            name, _ = parse_configured_surface(flow)
        except ValueError:
            continue  # a custom or Colang-defined rail declares no rewrite target
        surface = surfaces.get((RailDirection.OUTPUT, name))
        if surface is not None and surface.transform_target is not None:
            rewriting.append(flow)
    return rewriting


def relax_legacy_streaming_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Return ``config`` with legacy streaming defaults made valid for rewriting output rails.

    Returns ``config`` itself when nothing needs to change; never mutates its argument.
    """
    output = (config.get("rails") or {}).get("output") or {}
    streaming = output.get("streaming") or {}
    if not streaming.get("enabled"):
        return config

    rewriting = _output_flows_that_rewrite_response(output.get("flows") or [])
    if not rewriting:
        return config

    relaxed = dict(streaming)
    if relaxed.get("stream_first", True):
        relaxed["stream_first"] = False
    if relaxed.get("context_size", _LEGACY_CONTEXT_SIZE) == _LEGACY_CONTEXT_SIZE:
        relaxed["context_size"] = 0
    if relaxed == streaming:
        return config

    logger.warning(
        "Output rails %s rewrite the response, so streaming runs with stream_first=false and context_size=0 "
        "instead of the defaults saved with this config, which nemoguardrails no longer accepts",
        rewriting,
    )
    return {**config, "rails": {**config["rails"], "output": {**output, "streaming": relaxed}}}
