# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A trial's spans land in one trace derived from its session, and a caller's ``traceparent`` is read per W3C."""

from __future__ import annotations

import hashlib

from nemo_evals.gateway.policy import ModelAccess
from nemo_evals.gateway.spans import MAX_TRACESTATE_LENGTH, CallerSpan, trial_root_span_id, trial_trace_id
from nemo_evals.gateway.tokens import TrialClaims

TRIAL = TrialClaims(session="run-1:hello__abc", intake_workspace="evals", model_access=ModelAccess.OPEN)
CALLER_TRACE = "01a182ecbd457d118e02830498804285"
CALLER_SPAN = "9fe2c6bed0c1c48c"


def test_the_trace_and_root_span_are_derived_from_the_session() -> None:
    digest = hashlib.sha256(TRIAL.session.encode()).hexdigest()

    assert (trial_trace_id(TRIAL), trial_root_span_id(TRIAL)) == (digest[:32], digest[32:48])


def test_a_future_traceparent_version_is_read_by_its_version_00_fields() -> None:
    caller = CallerSpan.from_headers(f"01-{CALLER_TRACE}-{CALLER_SPAN}-01-future")

    assert caller == CallerSpan(CALLER_TRACE, CALLER_SPAN)


def test_an_oversized_tracestate_is_left_off() -> None:
    caller = CallerSpan.from_headers(f"00-{CALLER_TRACE}-{CALLER_SPAN}-01", "a=" + "b" * MAX_TRACESTATE_LENGTH)

    assert caller == CallerSpan(CALLER_TRACE, CALLER_SPAN)
