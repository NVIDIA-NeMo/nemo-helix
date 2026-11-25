# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply image-local Fabric compatibility fixes to every runtime process."""

from nemo_studio_assistant.fabric_compat import (
    apply_deepagents_mcp_env_compatibility,
)

apply_deepagents_mcp_env_compatibility()
