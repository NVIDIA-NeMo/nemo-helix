# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the inference-gateway upstream-requests counter and its labeling."""

from unittest.mock import Mock

import pytest
from nhx.core.inference_gateway.api import metrics
from nhx.core.inference_gateway.api.metrics import (
    METRIC_LABEL_UNKNOWN,
    UpstreamRequestLabels,
    record_upstream_request,
)


@pytest.fixture
def captured_counter(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the module-level counter with a mock and return it.

    The real counter binds to the global MeterProvider at import time (a no-op
    provider under test), so asserting on ``.add`` calls is the way to verify the
    emitted attributes without standing up a MeterProvider.
    """
    counter = Mock()
    monkeypatch.setattr(metrics, "upstream_requests_counter", counter)
    return counter


def _attributes(counter: Mock) -> dict[str, str]:
    counter.add.assert_called_once()
    args, kwargs = counter.add.call_args
    assert args[0] == 1
    return kwargs["attributes"]


def test_records_all_labels_for_a_passthrough_error(captured_counter: Mock) -> None:
    record_upstream_request(
        UpstreamRequestLabels(workspace="ws", model="ws/llama", model_provider="nim"),
        upstream_status=503,
        gateway_status=503,
    )
    assert _attributes(captured_counter) == {
        "workspace": "ws",
        "model": "ws/llama",
        "model_provider": "nim",
        "upstream_status": "503",
        "gateway_status": "503",
    }


def test_wrapped_rejection_records_distinct_upstream_and_gateway_status(captured_counter: Mock) -> None:
    # An upstream 401/403/404 is returned to the client as 424; both are recorded.
    record_upstream_request(
        UpstreamRequestLabels(workspace="ws", model="ws/llama", model_provider="nim"),
        upstream_status=401,
        gateway_status=424,
    )
    attrs = _attributes(captured_counter)
    assert attrs["upstream_status"] == "401"
    assert attrs["gateway_status"] == "424"


def test_success_records_matching_statuses(captured_counter: Mock) -> None:
    record_upstream_request(
        UpstreamRequestLabels(workspace="ws", model="ws/llama", model_provider="nim"),
        upstream_status=200,
        gateway_status=200,
    )
    attrs = _attributes(captured_counter)
    assert attrs["upstream_status"] == "200"
    assert attrs["gateway_status"] == "200"


def test_missing_upstream_status_uses_sentinel(captured_counter: Mock) -> None:
    # A pure network/internal failure: no upstream HTTP response was received.
    record_upstream_request(
        UpstreamRequestLabels(workspace="ws", model="ws/llama", model_provider="nim"),
        upstream_status=None,
        gateway_status=502,
    )
    attrs = _attributes(captured_counter)
    assert attrs["upstream_status"] == METRIC_LABEL_UNKNOWN
    assert attrs["gateway_status"] == "502"


def test_missing_routing_labels_use_sentinel(captured_counter: Mock) -> None:
    record_upstream_request(None, upstream_status=200, gateway_status=200)
    attrs = _attributes(captured_counter)
    assert attrs["workspace"] == METRIC_LABEL_UNKNOWN
    assert attrs["model"] == METRIC_LABEL_UNKNOWN
    assert attrs["model_provider"] == METRIC_LABEL_UNKNOWN


def test_partial_routing_labels_fill_only_missing_with_sentinel(captured_counter: Mock) -> None:
    record_upstream_request(
        UpstreamRequestLabels(workspace="ws", model=None, model_provider="nim"),
        upstream_status=429,
        gateway_status=429,
    )
    attrs = _attributes(captured_counter)
    assert attrs["workspace"] == "ws"
    assert attrs["model"] == METRIC_LABEL_UNKNOWN
    assert attrs["model_provider"] == "nim"


def test_gateway_status_is_always_a_concrete_string(captured_counter: Mock) -> None:
    # gateway_status is non-optional: the gateway always returns *some* status.
    record_upstream_request(UpstreamRequestLabels(), upstream_status=None, gateway_status=500)
    assert _attributes(captured_counter)["gateway_status"] == "500"
