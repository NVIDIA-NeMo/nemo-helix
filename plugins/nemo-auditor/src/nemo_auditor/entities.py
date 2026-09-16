# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auditor plugin entity definitions stored in the NeMo Helix entity store."""

from __future__ import annotations

from typing import Any

from nemo_helix_plugin.entity import NemoEntity
from pydantic import BaseModel, ConfigDict, Field, RootModel


class AuditClassConfig(RootModel[dict[str, Any]]):
    """Per-class plugin configuration mapping."""


class AuditModuleConfig(RootModel[dict[str, AuditClassConfig]]):
    """Per-module plugin configuration mapping."""


AuditRootPluginConfig = dict[str, AuditModuleConfig | AuditClassConfig]


class AuditSystemData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verbose: int = Field(default=0, ge=0, le=1)
    narrow_output: bool = False
    parallel_requests: bool = False
    parallel_attempts: bool | int = False
    lite: bool = True
    show_z: bool = False
    enable_experimental: bool = False


class AuditRunData(BaseModel):
    seed: int | None = None
    deprefix: bool = True
    eval_threshold: float = Field(default=0.5, ge=0, le=1)
    generations: int = Field(default=5, ge=1)
    probe_tags: str | None = None
    user_agent: str = "garak/{version} (LLM vulnerability scanner https://garak.ai)"


class AuditPluginsData(BaseModel):
    model_type: str | None = None
    model_name: str | None = None
    probe_spec: str = "all"
    detector_spec: str = "auto"
    extended_detectors: bool = False
    buff_spec: str | None = None
    buffs_include_original_prompt: bool = False
    buff_max: str | None = None
    detectors: AuditRootPluginConfig = Field(default_factory=dict)
    generators: AuditRootPluginConfig = Field(default_factory=dict)
    buffs: AuditRootPluginConfig = Field(default_factory=dict)
    harnesses: AuditRootPluginConfig = Field(default_factory=dict)
    probes: AuditRootPluginConfig = Field(default_factory=dict)


class AuditReportData(BaseModel):
    report_prefix: str = "run1"
    taxonomy: str | None = None
    report_dir: str = "garak_runs"
    show_100_pass_modules: bool = True


class AuditConfig(NemoEntity, entity_type="auditor_audit_config"):
    """Audit configuration stored in the entity store."""

    description: str | None = Field(default=None, description="Config description")
    system: AuditSystemData = Field(default_factory=AuditSystemData)
    run: AuditRunData = Field(default_factory=AuditRunData)
    plugins: AuditPluginsData = Field(default_factory=AuditPluginsData)
    reporting: AuditReportData = Field(default_factory=AuditReportData)


class AuditTarget(NemoEntity, entity_type="auditor_audit_target"):
    """Audit target (model under test) stored in the entity store."""

    description: str | None = Field(default=None, description="Target description")
    type: str = Field(..., description="Target type (e.g., 'nim', 'openai').")
    model: str = Field(..., description="Model identifier.")
    options: dict[str, Any] = Field(default_factory=dict, description="Additional target options.")


def get_entity_types() -> list[type[NemoEntity]]:
    """Return entity classes the auditor plugin registers with the entity store.

    Wired into the ``nemo.entities`` entry-point group so the entity service
    validates writes against these schemas before persisting them.
    """
    return [AuditConfig, AuditTarget]
