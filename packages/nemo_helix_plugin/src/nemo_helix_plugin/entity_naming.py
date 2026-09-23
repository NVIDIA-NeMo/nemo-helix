# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single definition of the entity name rule; ``nhx.common`` re-exports it.

Lives here because ``nhx_common`` depends on this package, not the reverse.
Models using ``pattern=NAME_PATTERN`` need ``regex_engine="python-re"``.
"""

# RFC 1035 compliant pattern with temporary support for special characters.
# Allows lowercase letters, digits, hyphens, and temporarily: @, ., +, _
# - Must start with a lowercase letter [a-z]
# - Length: 2-63 characters
# - No consecutive hyphens (--)
# - Must not end with a hyphen
# TODO(#3530): Remove @, ., +, _ once versioning is implemented and predefined target names (e.g., llama-3.2-3b-instruct@v1.0.0+A100) are updated.
NAME_PATTERN = r"^[a-z](?!.*--)[a-z0-9\-@.+_]{1,62}(?<!-)$"

NAME_MAX_LENGTH = 63

NAME_PATTERN_DESCRIPTION = (
    "Name must start with a lowercase letter, be 2-63 characters, "
    "and use lowercase letters, digits, hyphens, and dots "
    "(no consecutive hyphens, cannot end with a hyphen)."
)
