# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for entity validation."""

from nemo_helix_plugin.entity_naming import (
    NAME_MAX_LENGTH as NAME_MAX_LENGTH,
)
from nemo_helix_plugin.entity_naming import (
    NAME_PATTERN as NAME_PATTERN,
)
from nemo_helix_plugin.entity_naming import (
    NAME_PATTERN_DESCRIPTION as NAME_PATTERN_DESCRIPTION,
)

# Field length constraints
MAX_LENGTH_255 = 255

# Regex patterns for field validation
REGEX_WORD_CHARACTER_DOT_DASH = r"^[\w\-.]+$"
REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION = (
    "Allowed characters: letters (a-z, A-Z), digits (0-9), underscores, hyphens, and dots."
)
REGEX_WORD_CHARACTER_DOT_DASH_SLASH = r"^[\w\-./]+$"
REGEX_WORD_CHARACTER_DOT_DASH_OR_BLANK = r"^[\w\-.@:]*$"
REGEX_WORD_CHARACTER_DOT_DASH_OR_BLANK_OR_PLUS = r"^[\w\-\+.@:]*$"

# Special value to indicate all workspaces
ALL_WORKSPACES = "-"

# Default workspace when none is specified
DEFAULT_WORKSPACE = "default"

# System workspace used for platform-provided entities
SYSTEM_WORKSPACE = "system"
