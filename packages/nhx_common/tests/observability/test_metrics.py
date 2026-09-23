# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.observability.metrics import _format_metric_name


def test_format_metric_name():
    assert _format_metric_name("nhx", "secrets", "api_requests_total") == "nhx.secrets.api.requests.total"
    assert _format_metric_name("nhx", "", "api_requests_total") == "nhx.api.requests.total"
    assert _format_metric_name("nhx", "backend_service", "error_count") == "nhx.backend.service.error.count"
    assert (
        _format_metric_name("customnamespace", "customsubsystem", "custom_metric_name")
        == "customnamespace.customsubsystem.custom.metric.name"
    )
