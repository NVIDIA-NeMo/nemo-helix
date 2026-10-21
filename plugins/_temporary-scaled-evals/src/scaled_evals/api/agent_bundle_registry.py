# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from scaled_evals.api.auth import CurrentPrincipal
from scaled_evals.api.db import Database


def accessible_bundle_for_run(
    db: Database, principal: CurrentPrincipal, bundle_id: str, *, include_all: bool
) -> dict[str, Any] | None:
    row = db.agent_bundles.get_accessible(bundle_id, owner_id=principal.owner_id, include_all=include_all)
    if row is None:
        return None
    return {
        "bundle_id": row["id"],
        "bundle_name": row["bundle_name"],
        "agent_name": row["agent_name"],
        "agent_version": row["agent_version"],
        "image_ref": row["image_ref"],
        "image_digest": row["image_digest"],
        "entrypoint": row["entrypoint"],
        "platform": row["platform"],
        "runtime_abi": row["runtime_abi"],
        "bundle_layout_version": row["bundle_layout_version"],
        "builder_profile": row["builder_profile"],
        "source_lock_digest": row["source_lock_digest"],
        "fingerprint": row["fingerprint"],
        "visibility": row["visibility"],
        "qualification_status": row["qualification_status"],
        "qualification_evidence": row["qualification_evidence"],
        "metadata": row["metadata"],
    }
