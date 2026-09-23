# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

OutputFormat = Literal["table", "json", "yaml", "markdown", "csv", "raw"]  # Supported output formats
TimestampFormat = Literal["relative", "iso8601"]  # Supported timestamp formats
