# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway proxy metrics.

Follows the platform's established counter pattern (``metrics.get_meter`` +
``create_counter`` from ``nhx.common.observability``, as used by the jobs and
auth services). The MeterProvider and the Prometheus ``/metrics`` endpoint are
wired centrally by ``nhx_platform_runner`` for every hosted service, so creating
a counter here is all that is needed for it to be scraped — no per-service
exporter or endpoint wiring.

The single counter here records every upstream proxy attempt the gateway makes,
dimensioned so an operator can see *where* upstream rate-limits and outages are
coming from (which workspace, model, and model provider) and *what* the caller
actually received. Counting every attempt (2xx included) keeps the denominator
honest so an error *rate* can be computed in a dashboard
(``5xx / all``) rather than only an absolute error count — matching the RED /
Prometheus-apiserver / Envoy convention of one request counter dimensioned by
status, not a separate errors counter.
"""

from dataclasses import dataclass

from nhx.common.observability import create_counter
from opentelemetry import metrics

meter = metrics.get_meter(__name__)

# Sentinel label value for a dimension that is unknown in the current branch.
# Used for the upstream status on a pure network/internal failure (the gateway
# never received an HTTP response from upstream) and for any routing context a
# caller could not supply. Keeping a sentinel (rather than omitting the label)
# means every series carries the full attribute set, so the counter total stays
# a true count of all proxy attempts.
METRIC_LABEL_UNKNOWN = "unknown"

# One counter for every upstream proxy attempt, keyed by routing context and by
# both status codes. ``upstream.requests.total`` reads as "requests the gateway
# made to an upstream provider"; "requests" (not "responses") is the term
# operators expect for this RED-style metric.
upstream_requests_counter = create_counter(
    meter=meter,
    subsystem="inference_gateway",
    name="upstream.requests.total",
    description=(
        "Total upstream inference requests proxied by the gateway, labeled by workspace, "
        "model, model provider, the true upstream HTTP status, and the HTTP status the "
        "gateway returned to the client."
    ),
)


@dataclass(frozen=True)
class UpstreamRequestLabels:
    """Routing context for an upstream proxy attempt, used as metric attributes.

    Every field is optional so a caller that lacks full context (e.g. the mock
    path, or an internal caller) still records a well-formed series: a missing
    field becomes :data:`METRIC_LABEL_UNKNOWN` rather than dropping the label.
    """

    workspace: str | None = None
    model: str | None = None
    model_provider: str | None = None


def record_upstream_request(
    labels: UpstreamRequestLabels | None,
    upstream_status: int | None,
    gateway_status: int,
) -> None:
    """Increment the upstream-requests counter for one proxy attempt.

    Args:
        labels: Routing context (workspace, model, model provider). ``None`` and
            any missing field are recorded as :data:`METRIC_LABEL_UNKNOWN`.
        upstream_status: The true HTTP status the upstream provider returned, or
            ``None`` when the gateway never received an upstream response (a
            network error or an internal gateway error). ``None`` is recorded as
            :data:`METRIC_LABEL_UNKNOWN`.
        gateway_status: The HTTP status the gateway returns to the client. This
            differs from *upstream_status* on the wrapped-rejection path (an
            upstream 401/403/404 is returned to the client as 424) and on the
            network/internal-error paths (502 / 500), and equals it for the
            pass-through 429/5xx and 2xx cases.
    """
    labels = labels or UpstreamRequestLabels()
    upstream_requests_counter.add(
        1,
        attributes={
            "workspace": labels.workspace or METRIC_LABEL_UNKNOWN,
            "model": labels.model or METRIC_LABEL_UNKNOWN,
            "model_provider": labels.model_provider or METRIC_LABEL_UNKNOWN,
            "upstream_status": str(upstream_status) if upstream_status is not None else METRIC_LABEL_UNKNOWN,
            "gateway_status": str(gateway_status),
        },
    )
