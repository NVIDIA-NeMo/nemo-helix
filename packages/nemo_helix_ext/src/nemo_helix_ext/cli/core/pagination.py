# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-exports; the pagination helpers live in :mod:`nemo_helix_plugin.cli_pagination`."""

from nemo_helix_plugin.cli_pagination import AllCursorPagesResponse as AllCursorPagesResponse
from nemo_helix_plugin.cli_pagination import AllPagesResponse as AllPagesResponse
from nemo_helix_plugin.cli_pagination import CursorPageResponse as CursorPageResponse
from nemo_helix_plugin.cli_pagination import OffsetPageResponse as OffsetPageResponse
from nemo_helix_plugin.cli_pagination import PaginationType as PaginationType
from nemo_helix_plugin.cli_pagination import collect_cursor_pages as collect_cursor_pages
from nemo_helix_plugin.cli_pagination import collect_offset_pages as collect_offset_pages
from nemo_helix_plugin.cli_pagination import collect_pages as collect_pages
from nemo_helix_plugin.cli_pagination import warn_if_more_pages as warn_if_more_pages
