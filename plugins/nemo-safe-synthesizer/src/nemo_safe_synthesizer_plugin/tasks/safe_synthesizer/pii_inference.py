# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point Safe Synthesizer's PII inference at the job's model provider."""

from dataclasses import dataclass

from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

# Matches the auth proxy's 300s read timeout; reasoning models outlast NSS's 60s default.
PII_INFERENCE_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class PiiInference:
    """The model provider route and model that Safe Synthesizer's PII inference calls."""

    endpoint_url: str
    model_id: str


def with_pii_inference(ss: SafeSynthesizer, pii_inference: PiiInference | None) -> SafeSynthesizer:
    """Return ``ss`` sending PII inference to ``pii_inference``, or unchanged when there is none."""
    if pii_inference is None:
        return ss
    return ss.with_inference(  # ty: ignore[unresolved-attribute]
        pii_inference.endpoint_url,
        pii_inference.model_id,
        timeout_seconds=PII_INFERENCE_TIMEOUT_SECONDS,
    )
