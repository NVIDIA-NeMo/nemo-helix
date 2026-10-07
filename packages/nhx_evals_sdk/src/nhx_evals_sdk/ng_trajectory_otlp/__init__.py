# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project NeMo-Gym rollout records onto OTLP spans.

Self-contained on purpose: nothing here imports the rest of this SDK, and its only dependency is
``opentelemetry-proto``, so the projection can be lifted into its own distribution if a producer
outside the evaluator ever wants it. Keep it that way.
"""

from nhx_evals_sdk.ng_trajectory_otlp.convert import SPAN_KIND_ATTRIBUTE, rollout_to_resource_spans

__all__ = [
    "SPAN_KIND_ATTRIBUTE",
    "rollout_to_resource_spans",
]
