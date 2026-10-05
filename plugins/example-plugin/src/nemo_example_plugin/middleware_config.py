# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Middleware config entity for the example inference middleware plugin.

Stores the configuration for :class:`~nemo_example_plugin.middleware.ExampleInferenceMiddleware`.
A ``MiddlewareCall`` in a VirtualModel can reference this entity via ``config_id``
(``"workspace/my-filter-config"``) instead of embedding the config inline.

Referencing by ``config_id`` means:

- The config is stored and versioned in the entity store.
- Multiple VirtualModels can share the same config.
- Updating the config entity triggers IGW to re-resolve and cache the new config
  for every VirtualModel that references it — no VirtualModel edit required.
- The config must have its own CRUD API (see
  :mod:`nemo_example_plugin.middleware_service`) so operators can create and
  update it.

If all your configs are simple and don't need versioning or sharing, use inline
config (``MiddlewareCall.config``) and skip the entity + CRUD API entirely.
"""

from __future__ import annotations

from nemo_helix_plugin.entity import NemoEntity


class ExampleMiddlewareConfig(NemoEntity, entity_type="example_middleware_config"):
    """Keyword content-filter configuration stored in the entity store.

    References to this entity in ``MiddlewareCall.config_id`` are resolved by
    :meth:`~nemo_example_plugin.middleware.ExampleInferenceMiddleware.get_middleware_config`
    on every polling cycle so updates propagate automatically.

    Attributes:
        blocked_keywords: Case-insensitive list of keywords.  Requests containing
            any of these in the message content are blocked.  Responses containing
            them have the keyword replaced with ``"[REDACTED]"``.
        block_message: Human-readable refusal message returned to the caller when
            a request is blocked.  Included in the response body.
    """

    blocked_keywords: list[str] = []
    """Case-insensitive keywords that trigger request blocking and response redaction."""

    block_message: str = "Your request contains content that is not permitted."
    """Message returned to the caller when a request is blocked."""
