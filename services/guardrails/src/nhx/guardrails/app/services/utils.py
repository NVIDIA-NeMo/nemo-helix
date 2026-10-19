# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from nhx.common.entities.utils import parse_entity_ref


def normalize_config_ids(config_ids: List[str], default_workspace: str) -> List[str]:
    """Returns list of config IDs as fully-qualified entity references.

    If a config ID does not include a workspace prefix, it uses the provided default workspace.

    Args:
        config_ids: List of config IDs, which may or may not include a workspace prefix.
        default_workspace: Workspace to use as the prefix for unqualified IDs.

    Returns:
        A new list where every config ID is in `workspace/name` format.

    Raises:
        ValueError: If any config ID is malformed
    """
    result = []
    for config_id in config_ids:
        ref = parse_entity_ref(config_id, default_workspace=default_workspace)
        result.append(f"{ref.workspace}/{ref.name}")

    return result
