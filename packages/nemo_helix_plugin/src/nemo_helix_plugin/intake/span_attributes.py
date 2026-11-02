# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span attributes that producers and Intake agree on, so a model call's usage is counted once per session.

See "Counting model usage once" in ``services/intake/README.md``.
"""

GATEWAY_CALL_ATTRIBUTE = "nemo.inference_gateway.call"
"""``True`` on the span the Inference Gateway records for one upstream model call it served, after routing.

Set it on one leaf span per billed call, carrying the usage the provider reported. When the caller sent a W3C
``traceparent``, put the span under the caller's span, in the caller's trace: Intake then counts it instead of the
caller's spans for the same call.
"""

USAGE_SUPERSEDED_ATTRIBUTE = "nemo.usage.superseded_by_gateway"
"""``True`` on the root span of an agent's trace whose model calls the gateway recorded in another trace of the session.

Set it only when every model call the trace reports went through a gateway model that records calls. For each metric
the gateway's spans report, Intake then stops adding this trace's usage to the session. The trace keeps its own totals,
and a metric the gateway doesn't report still counts from it.
"""
