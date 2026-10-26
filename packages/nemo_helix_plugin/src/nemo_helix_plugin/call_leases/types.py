# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call leases: which trial a sandbox's network address belongs to, and for how long.

A platform service that creates a sandbox leases the sandbox's pod IP to the sandbox's trial for the window it
holds that IP. The Inference Gateway records model calls and accepts agent telemetry from a leased address into
that trial's session and trace, so a sandbox never names its own trial.
"""

from __future__ import annotations

import ipaddress
import re
from datetime import datetime
from typing import Literal, Self, TypedDict

from nemo_helix_plugin.entity import NemoEntity
from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator

_TRACE_ID = re.compile(r"[0-9a-f]{32}")
_SPAN_ID = re.compile(r"[0-9a-f]{16}")

CallLeaseRole = Literal["agent", "verifier"]


def _ip(value: str) -> str:
    return str(ipaddress.ip_address(value))


def _hex_id(value: str, pattern: re.Pattern[str], what: str) -> str:
    if not pattern.fullmatch(value) or not value.strip("0"):
        raise ValueError(f"{what} must be lowercase hex ({pattern.pattern}) and not all zeros")
    return value


class _CallLeaseFields(BaseModel):
    client_address: str = Field(description="The sandbox pod's IP address while the lease holds.")
    session_id: str = Field(min_length=1, max_length=1024, description="The trial's Intake session.")
    trace_id: str = Field(description="The trial's trace id: 32 lowercase hex characters.")
    root_span_id: str = Field(description="The trial's root span id: 16 lowercase hex characters.")
    role: CallLeaseRole = Field(description="Which of the trial's sandboxes holds the address.")
    started_at: AwareDatetime = Field(description="When the sandbox started holding the address.")
    evaluation_name: str | None = Field(default=None, description="The Intake evaluation the trial belongs to.")
    test_case_name: str | None = Field(default=None, description="The Intake test case the trial runs.")

    @field_validator("client_address")
    @classmethod
    def _valid_ip(cls, value: str) -> str:
        return _ip(value)

    @field_validator("trace_id")
    @classmethod
    def _valid_trace_id(cls, value: str) -> str:
        return _hex_id(value, _TRACE_ID, "trace_id")

    @field_validator("root_span_id")
    @classmethod
    def _valid_root_span_id(cls, value: str) -> str:
        return _hex_id(value, _SPAN_ID, "root_span_id")


class CreateCallLeaseRequest(_CallLeaseFields):
    """Lease ``client_address`` to a trial from ``started_at`` until the lease is ended."""

    name: str = Field(description="Lease name, unique in the workspace; one lease per sandbox.")


class EndCallLeaseRequest(BaseModel):
    """End a lease: the sandbox stops holding its address at ``ended_at``."""

    ended_at: AwareDatetime


class CallLease(_CallLeaseFields, NemoEntity, entity_type="inference_call_lease"):
    """A sandbox address leased to one trial for ``[started_at, ended_at]``."""

    ended_at: AwareDatetime | None = Field(default=None, description="When the sandbox stopped holding the address.")

    @model_validator(mode="after")
    def _ends_after_it_starts(self) -> Self:
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at must not be before started_at")
        return self

    def holds(self, at: datetime) -> bool:
        """Whether the lease covers ``at``."""
        return self.started_at <= at and (self.ended_at is None or at <= self.ended_at)


class ListCallLeasesQueryParams(TypedDict, total=False):
    page: int
    page_size: int
    sort: str
    filter: str
