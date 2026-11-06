# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-profile checks that happen before any job is submitted."""

from __future__ import annotations

from dataclasses import dataclass

from config import Step

_STOCK_IMAGE = "nhx-tasks"
_ACCEPTED_BACKENDS = ("kubernetes_job", "docker")


class ProfileError(ValueError):
    """The execution profile cannot run this tool."""


@dataclass(frozen=True)
class ProfileCandidate:
    name: str
    provider: str
    backend: str
    task_image: str | None
    ttl_seconds_before_active: int
    ttl_seconds_active: int


@dataclass(frozen=True)
class ProfileFacts:
    name: str
    provider: str
    backend: str
    task_image: str | None
    ttl_seconds_before_active: int
    ttl_seconds_active: int


def choose_profile(candidates: list[ProfileCandidate], selector: str) -> ProfileFacts:
    """Use ``default``, a profile name, or ``provider/profile`` such as ``cpu/gpu``."""
    provider, name = _selector(selector)
    if provider is None:
        named = [item for item in candidates if item.name == name]
    else:
        named = [item for item in candidates if item.provider == provider and item.name == name]
    if not named:
        raise ProfileError(f"profile {selector!r} was not found. Registered: {_listing(candidates)}")
    subprocess_names = {item.name for item in candidates if item.backend == "subprocess"}
    if name in subprocess_names and provider in (None, "cpu"):
        alternatives = [
            item
            for item in candidates
            if item.backend in _ACCEPTED_BACKENDS and item.name not in subprocess_names and _stock_image(item)
        ]
        raise ProfileError(
            f"profile {selector!r} is registered for subprocess, so a CPU job using it runs on the host. "
            f"Pass --job-profile to select another profile. Available: {_listing(alternatives)}"
        )
    accepted = [item for item in named if item.backend in _ACCEPTED_BACKENDS]
    if not accepted:
        raise ProfileError(f"profile {selector!r} backend is {named[0].backend!r}; expected kubernetes_job or docker")
    if provider is None:
        cpu = [item for item in accepted if item.provider == "cpu"]
        pool = cpu or accepted
    else:
        pool = accepted
    chosen = _one(pool)
    if not _stock_image(chosen):
        raise ProfileError(f"profile {chosen.name!r} image {chosen.task_image!r} is not {_STOCK_IMAGE}")
    return ProfileFacts(
        name=chosen.name,
        provider=chosen.provider,
        backend=chosen.backend,
        task_image=chosen.task_image,
        ttl_seconds_before_active=chosen.ttl_seconds_before_active,
        ttl_seconds_active=chosen.ttl_seconds_active,
    )


def runnable_steps(
    chosen: list[tuple[Step, bool]],
    *,
    auth_enabled: bool,
    only: str | None,
) -> tuple[list[tuple[Step, bool]], list[str]]:
    """Drop auth steps when auth is off. An explicit ``--step`` of one of them is an error."""
    if auth_enabled:
        return chosen, []
    skipped = [step.id for step, _force in chosen if step.service == "auth"]
    if only is not None and skipped:
        raise ProfileError(f"step {only} calls the auth service, and auth is disabled on this platform")
    kept = [(step, force) for step, force in chosen if step.service != "auth"]
    return kept, skipped


def validate_profile(facts: ProfileFacts, steps: list[Step]) -> None:
    """Reject a profile whose task image is not the stock tasks image, or a step longer than its TTL."""
    if facts.backend not in _ACCEPTED_BACKENDS:
        raise ProfileError(f"profile {facts.name!r} backend is {facts.backend!r}; expected kubernetes_job or docker")
    if facts.task_image is not None and _STOCK_IMAGE not in facts.task_image:
        raise ProfileError(f"profile {facts.name!r} image {facts.task_image!r} is not {_STOCK_IMAGE}")
    for step in steps:
        if step.duration_seconds is not None and step.duration_seconds > facts.ttl_seconds_active:
            raise ProfileError(
                f"step {step.id} duration {step.duration_seconds}s exceeds "
                f"ttl_seconds_active {facts.ttl_seconds_active}s"
            )
        if step.timeout_seconds is not None and step.timeout_seconds > facts.ttl_seconds_active:
            raise ProfileError(
                f"step {step.id} timeout {step.timeout_seconds}s exceeds ttl_seconds_active {facts.ttl_seconds_active}s"
            )


def _one(candidates: list[ProfileCandidate]) -> ProfileCandidate:
    for backend in _ACCEPTED_BACKENDS:
        matches = [item for item in candidates if item.backend == backend]
        if len(matches) > 1:
            raise ProfileError(
                f"profile {matches[0].name!r} matches more than one {backend} profile: {_listing(matches)}"
            )
        if len(matches) == 1:
            return matches[0]
    raise ProfileError(f"profile {candidates[0].name!r} has no docker or kubernetes_job profile")


def _selector(value: str) -> tuple[str | None, str]:
    if "/" not in value:
        return None, value
    provider, name = value.split("/", 1)
    if not provider or not name or "/" in name:
        raise ProfileError(f"profile {value!r} must be a name or provider/profile")
    return provider, name


def _stock_image(candidate: ProfileCandidate) -> bool:
    return candidate.task_image is None or _STOCK_IMAGE in candidate.task_image


def _listing(candidates: list[ProfileCandidate]) -> str:
    if not candidates:
        return "none"
    labels = sorted(f"{item.provider}/{item.name} ({item.backend})" for item in candidates)
    return ", ".join(labels)
