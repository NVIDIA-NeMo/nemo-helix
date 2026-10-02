# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-facing Intake readiness guidance."""

CLICKHOUSE_UNAVAILABLE_MESSAGE = (
    "ClickHouse storage is inaccessible. Check that ClickHouse is running, verify the configured URL and "
    "credentials, and ensure its data volume is mounted with readable and writable permissions."
)
