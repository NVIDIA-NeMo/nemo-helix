# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset loading and template utilities for evals SDK runtime."""

from nhx_evals_sdk.datasets.loader import (
    DatasetLoadError,
    discover_files,
    is_glob_pattern,
    load_dataset,
    load_dataset_as_dicts,
    load_file,
)
from nhx_evals_sdk.templates import (
    render_request,
    render_template,
)

__all__ = [
    "DatasetLoadError",
    "is_glob_pattern",
    "discover_files",
    "load_file",
    "load_dataset",
    "load_dataset_as_dicts",
    "render_request",
    "render_template",
]
