# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from nhx.common.observability import BaseContext


@dataclass
class SecretsContext(BaseContext):
    """
    Secrets context that will be enriched into logs and traces
    """

    otel_prefix: str = "secret"

    name: str | None = None
    namespace: str | None = None
