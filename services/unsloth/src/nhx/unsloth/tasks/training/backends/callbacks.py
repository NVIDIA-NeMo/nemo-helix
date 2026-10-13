# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training progress callbacks for Unsloth Jobs-service reporting.

Thin subclass of the shared
:class:`nhx.customization_common.training.callbacks.TrainingProgressCallback`
that stamps ``backend="unsloth"`` on each report by default.
"""

from typing import ClassVar

from nhx.customization_common.training.callbacks import (
    TrainingProgressCallback as _BaseTrainingProgressCallback,
)

__all__ = ["TrainingProgressCallback"]


class TrainingProgressCallback(_BaseTrainingProgressCallback):
    """Report Unsloth training progress to the Jobs service."""

    _default_backend: ClassVar[str | None] = "unsloth"
