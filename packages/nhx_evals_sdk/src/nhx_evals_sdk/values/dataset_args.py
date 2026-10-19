# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The union of dataset forms accepted as an evaluation ``dataset`` argument.

This union spans ``DatasetInput`` and ``BeirDataset``, which live in sibling modules, so it
belongs to neither. It cannot live in the ``values`` barrel either: that surface is lazy, while a
``|`` union needs the real classes at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from nhx_evals_sdk.retrieval.beir import BeirDataset
from nhx_evals_sdk.values.datasets import DatasetInput

__all__ = ["DatasetArg"]

DatasetArg: TypeAlias = DatasetInput | str | Path | BeirDataset
