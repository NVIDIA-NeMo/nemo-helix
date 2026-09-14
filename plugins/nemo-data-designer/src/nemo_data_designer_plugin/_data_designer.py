# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import data_designer.config as dd
import data_designer.interface.data_designer as data_designer_interface
from data_designer.interface.data_designer import DataDesigner
from data_designer_nemo.context import DataDesignerExecutionContext


def create_data_designer(
    *,
    artifact_path: Path | str,
    model_providers: list[dd.ModelProvider],
    dd_ctx: DataDesignerExecutionContext,
) -> DataDesigner:
    """Create the library interface without letting it reconfigure process logging."""
    data_designer_interface.configure_logging = _noop_configure_logging  # ty: ignore[invalid-assignment]
    return DataDesigner(
        artifact_path=artifact_path,
        model_providers=model_providers,
        secret_resolver=dd_ctx.get_secret_resolver(),
        seed_readers=dd_ctx.get_seed_readers(),
        person_reader=dd_ctx.get_person_reader(),
    )


def _noop_configure_logging(*args: object, **kwargs: object) -> None:
    pass
