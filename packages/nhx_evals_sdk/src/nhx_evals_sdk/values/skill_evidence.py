# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental adapter inputs for skill artifact scans and observed tool behavior."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from nhx_evals_sdk.values.skills import SkillSnapshot

ScanStatus = Literal["completed", "partial", "failed", "unavailable"]
FindingSeverity = Literal["critical", "high", "medium", "low", "info"]


class SkillContentBundle(BaseModel):
    """Original UTF-8 files associated with one skill revision."""

    model_config = ConfigDict(extra="forbid", strict=True)

    artifact_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    files: dict[str, str]

    @model_validator(mode="after")
    def canonical_paths(self) -> Self:
        SkillSnapshot.validate_files(list(self.files))
        return self


class SkillFinding(BaseModel):
    """A redacted finding supplied by a scanner or verifier."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rule: str = Field(min_length=1)
    category: str = Field(min_length=1)
    severity: FindingSeverity
    file: str | None = None
    line: int | None = Field(default=None, ge=1)


class SkillScanReport(BaseModel):
    """Scan evidence for an artifact digest or trial ID."""

    model_config = ConfigDict(extra="forbid", strict=True)

    subject_id: str = Field(min_length=1)
    scanner: str = Field(min_length=1)
    scanner_version: str = Field(min_length=1)
    status: ScanStatus
    findings: list[SkillFinding] = Field(default_factory=list)
    files_scanned: int = Field(ge=0)
    files_expected: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_coverage(self) -> Self:
        if self.files_expected is not None and self.files_scanned > self.files_expected:
            raise ValueError("files_scanned exceeds files_expected")
        if self.status == "completed" and self.files_expected is not None and self.files_scanned != self.files_expected:
            raise ValueError("a completed scan must cover every expected file")
        return self


class SkillTestReport(BaseModel):
    """Bundled test results produced by an isolated test runner."""

    model_config = ConfigDict(extra="forbid", strict=True)

    artifact_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: ScanStatus
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: int = Field(ge=0)
    skipped: int = Field(ge=0)
    coverage: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class SkillToolCall(BaseModel):
    """A tool invocation. Calls at the same step are unordered."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    step: int = Field(ge=0)
    tool: str = Field(min_length=1)
    component: str | None = Field(default=None, description="Canonical identity such as Skill:search or MCP:docs/read.")
    arguments: dict[str, JsonValue] | None = Field(
        default=None, description="Null means unavailable. An empty object records a call with no arguments."
    )
    observation: JsonValue | None = None
    succeeded: bool | None = None
    script: str | None = None
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    retry_of: str | None = None
    fault_owner: Literal["skill", "agent", "unknown"] = "unknown"
    productivity: Literal["productive", "wasted", "neutral"] | None = None


class SkillToolTrace(BaseModel):
    """Ordered tool evidence with explicit event coverage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    trial_id: str = Field(min_length=1)
    complete: bool = Field(description="True when all relevant events were captured. Individual fields may be unknown.")
    calls: list[SkillToolCall]
    initial_context: list[JsonValue] | None = Field(
        default=None, description="All preexisting context relevant to value handoffs; null means coverage is unknown."
    )

    @model_validator(mode="after")
    def validate_order_and_retries(self) -> Self:
        previous: dict[str, SkillToolCall] = {}
        last_step = -1
        for call in self.calls:
            if call.id in previous or call.step < last_step:
                raise ValueError("calls must have unique IDs and nondecreasing steps")
            if call.retry_of is not None:
                failed = previous.get(call.retry_of)
                if failed is None or failed.step >= call.step or failed.succeeded is not False:
                    raise ValueError("retry_of must identify a failed call in an earlier step")
            previous[call.id] = call
            last_step = call.step
        return self
