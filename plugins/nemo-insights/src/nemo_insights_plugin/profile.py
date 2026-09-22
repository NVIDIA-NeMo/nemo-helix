# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analysis-owned view of an optimizer.yaml agent profile."""

from pathlib import Path

from nemo_insights_plugin.contracts.profile import load_profile_model, resolve_ethos_path
from pydantic import BaseModel, ConfigDict, Field


class AnalysisProfile(BaseModel):
    """Only fields consumed by ``nemo insights``; all other keys are tolerated."""

    model_config = ConfigDict(extra="ignore")

    agent: str = Field(min_length=1)
    ethos: str | None = None
    workspace: str | None = None
    profile_dir: Path


def load_profile(path: Path) -> AnalysisProfile:
    """Load the analysis-owned subset of optimizer.yaml."""
    return load_profile_model(path, AnalysisProfile)


def pick_ethos(profile: AnalysisProfile) -> Path | None:
    """Resolve the profile's configured or conventional Ethos content."""
    return resolve_ethos_path(profile.profile_dir, profile.ethos)
