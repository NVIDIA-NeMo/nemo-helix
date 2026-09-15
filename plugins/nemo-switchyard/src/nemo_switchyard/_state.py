# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-level state mapping VirtualModels to registered Switchyard factories.

NOTE: Module-level state is necessary because the nemo_platform_plugin interface passes only
config_type in middleware_config (not the full config or factory_name), so we need
a lookup keyed by (vm_key, config_type, phase). IGW has its own caching layer but
doesn't expose it for this purpose. State is cleared on shutdown to prevent stale
entries on reload.

Phase tagging: each switchyard config is associated with the phase the user listed
it under — "request" if it's in `request_middleware`, "response" if in
`response_middleware`. process_request looks up only request-phase entries;
process_response only response-phase entries. Listing the same config under both
lists registers two phase entries pointing at the same factory — the user is
explicitly opting into both pipelines.

Concurrency caveat: check-then-set in on_virtual_model_upserted is not atomic.
Multiple parallel upserts of identical configs could both miss and attempt register().
For now, assumes single-threaded startup; revisit if concurrency becomes an issue.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

Phase = Literal["request", "response"]

# config_hash -> factory_name. Multiple VMs with identical configs share one factory.
FACTORIES_BY_CONFIG_HASH: dict[str, str] = {}

# vm_id -> [config_hash, ...]. A VM may have multiple nemo-switchyard middlewares
# (e.g., random_routing + translate); we track all so destroy can clean them all up.
VM_CONFIG_MAPPING: dict[str, list[str]] = {}

# (vm_key, config_type, phase) -> config_hash. Phase is "request" or "response".
# A single VM can have the same config_type registered for both phases (the user
# listed it under both request_middleware and response_middleware).
VM_NAME_TO_CONFIG_HASH: dict[tuple[str, str, Phase], str] = {}

# config_hash -> native Algorithm binding. In-process per IGW replica; not shared
# across replicas. Cleared on shutdown and when the last VM using the hash is destroyed.
NATIVE_BY_CONFIG_HASH: dict[str, Any] = {}


def config_hash(config: dict[str, Any], config_type: str) -> str:
    """Compute a deterministic hash of (config_type, config) for factory dedup."""
    data = {"config_type": config_type, "config": config}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


def clear_all() -> None:
    """Clear all state. Called on shutdown to prevent stale entries on reload."""
    FACTORIES_BY_CONFIG_HASH.clear()
    VM_CONFIG_MAPPING.clear()
    VM_NAME_TO_CONFIG_HASH.clear()
    NATIVE_BY_CONFIG_HASH.clear()
