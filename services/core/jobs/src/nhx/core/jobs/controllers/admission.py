# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Protocol

from nemo_helix_plugin.jobs.types import HelixJobStepWithContext


@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of an admission check for a ready job step.

    ``reason`` is a human-readable explanation recorded in the step's ``status_details``
    when the step is not admitted.
    """

    admitted: bool
    reason: str | None = None

    @classmethod
    def admit(cls) -> "AdmissionDecision":
        return cls(admitted=True)

    @classmethod
    def defer(cls, reason: str) -> "AdmissionDecision":
        return cls(admitted=False, reason=reason)


class AdmissionController(Protocol):
    """Decides whether a ready job step may be dispatched to a backend now.

    Called by the scheduler once per ready step, in FIFO (``created_at``) order. A step that is
    not admitted is queued and reconsidered on a later scheduling pass.
    """

    def check(self, step: HelixJobStepWithContext) -> AdmissionDecision: ...


class AlwaysAdmit:
    """Admission controller that admits every step; preserves pre-queueing behavior."""

    def check(self, step: HelixJobStepWithContext) -> AdmissionDecision:
        return AdmissionDecision.admit()
