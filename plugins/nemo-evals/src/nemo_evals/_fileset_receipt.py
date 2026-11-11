# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity retained from a successful Files allocation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilesetReceipt:
    ref: str
    fileset_id: str
