# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-data path resolution for NeMo Helix local state.

Re-exports from :mod:`nemo_helix_plugin.config` — the canonical
implementation now lives in the plugin package.
"""

from nemo_helix_plugin.config import nhx_user_data_dir as nhx_user_data_dir

# Keep the env var constants exported for backward compat.
NHX_DATA_DIR_ENV_VAR = "NHX_DATA_DIR"
XDG_DATA_HOME_ENV_VAR = "XDG_DATA_HOME"
