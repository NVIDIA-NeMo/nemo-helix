# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-profile checks that happen before any job is submitted."""

from __future__ import annotations

from dataclasses import dataclass

from config import Step

_STOCK_IMAGE = "nhx-tasks"


class ProfileError(ValueError):
    """The saved context is not a Kubernetes profile this tool can use."""


@dataclass(frozen=True)
class ProfileFacts:
    name: str
    backend: str
    task_image: str | None
    ttl_seconds_before_active: int
    ttl_seconds_active: int


def validate_profile(facts: ProfileFacts, steps: list[Step]) -> None:
    """Accept only a Kubernetes profile whose task image is the stock tasks image."""
    if facts.backend != "kubernetes_job":
        raise ProfileError(f"profile {facts.name!r} backend is {facts.backend!r}; expected kubernetes_job")
    if facts.task_image is not None and _STOCK_IMAGE not in facts.task_image:
        raise ProfileError(f"profile {facts.name!r} image {facts.task_image!r} is not {_STOCK_IMAGE}")
    for step in steps:
        if step.duration_seconds is not None and step.duration_seconds > facts.ttl_seconds_active:
            raise ProfileError(
                f"step {step.id} duration {step.duration_seconds}s exceeds "
                f"ttl_seconds_active {facts.ttl_seconds_active}s"
            )
