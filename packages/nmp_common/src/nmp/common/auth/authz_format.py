# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-exported from nemo_platform_plugin.authz_format. Import from there instead."""

from nemo_platform_plugin.authz_format import (
    NMP_SCOPE_PATTERN as NMP_SCOPE_PATTERN,
)
from nemo_platform_plugin.authz_format import (
    PERMISSION_ID_PATTERN as PERMISSION_ID_PATTERN,
)
from nemo_platform_plugin.authz_format import (
    is_valid_nmp_scope_id as is_valid_nmp_scope_id,
)
from nemo_platform_plugin.authz_format import (
    is_valid_permission_id as is_valid_permission_id,
)
from nemo_platform_plugin.authz_format import (
    is_wildcard_permission as is_wildcard_permission,
)
from nemo_platform_plugin.authz_format import (
    looks_like_mistaken_permission_for_scope as looks_like_mistaken_permission_for_scope,
)
from nemo_platform_plugin.authz_format import (
    looks_like_mistaken_scope_for_permission as looks_like_mistaken_scope_for_permission,
)
from nemo_platform_plugin.authz_format import (
    validate_nmp_scope_strings_for_config as validate_nmp_scope_strings_for_config,
)
from nemo_platform_plugin.authz_format import (
    validate_permission_strings as validate_permission_strings,
)
from nemo_platform_plugin.authz_format import (
    validate_runtime_authorize_scopes as validate_runtime_authorize_scopes,
)
from nemo_platform_plugin.authz_format import (
    validate_static_authz_data as validate_static_authz_data,
)
