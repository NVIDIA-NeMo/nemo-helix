# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from config import Step
from job_profile import ProfileError, ProfileFacts, validate_profile


def _facts(backend: str, image: str | None, active: int) -> ProfileFacts:
    return ProfileFacts(
        name="default",
        backend=backend,
        task_image=image,
        ttl_seconds_before_active=60,
        ttl_seconds_active=active,
    )


def test_kubernetes_profile_with_stock_image_is_accepted() -> None:
    step = Step("idle-01", "idle", 1, 300, None, None)
    validate_profile(_facts("kubernetes_job", None, 3600), [step])
    validate_profile(_facts("kubernetes_job", "registry.example/nhx-tasks:latest", 3600), [step])


def test_non_kubernetes_backend_is_rejected() -> None:
    step = Step("exit-01", "exit", 1, None, None, None)
    with pytest.raises(ProfileError, match="kubernetes_job"):
        validate_profile(_facts("volcano_job", None, 3600), [step])


def test_custom_image_is_rejected() -> None:
    step = Step("exit-01", "exit", 1, None, None, None)
    with pytest.raises(ProfileError, match="nhx-tasks"):
        validate_profile(_facts("kubernetes_job", "python:3.12", 3600), [step])


def test_duration_beyond_active_ttl_is_rejected() -> None:
    step = Step("idle-01", "idle", 1, 300, None, None)
    with pytest.raises(ProfileError, match="ttl_seconds_active"):
        validate_profile(_facts("kubernetes_job", None, 60), [step])
