# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extended filesets module with FilesetFileSystem support.

This module provides high-level file operations (upload, download, etc.) and
fsspec integration for NeMo Platform filesets via the sdk.files.fsspec property.

Located at: nemo_platform/filesets/ (after vendoring)
"""

from .filesystem.callbacks import RichFileProgressCallback as RichFileProgressCallback
from .filesystem.callbacks import RichProgressCallback as RichProgressCallback
from .filesystem.filesystem import AsyncFilesetFileSystem as AsyncFilesetFileSystem
from .filesystem.filesystem import FilesetFileSystem as FilesetFileSystem
from .filesystem.filesystem import FilesetPathError as FilesetPathError
from .filesystem.filesystem import build_fileset_ref as build_fileset_ref
from .filesystem.filesystem import parse_fileset_path as parse_fileset_path
from .filesystem.filesystem import parse_fileset_ref as parse_fileset_ref
from .resources import ListFilesResponse as ListFilesResponse
