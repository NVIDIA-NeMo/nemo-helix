# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-stamping helper for ``HelixJobSpec``.

:func:`stamp_profile` walks ``spec.steps`` and applies ``step.executor.profile``
when the compiler didn't set one explicitly. This implements the phase-1
profile-resolution invariant (see plan-phase-1.md §4b) — per-step overrides
win; otherwise the job-level profile is stamped across every step.

Plugin services call this inside their custom ``compile()`` right before
returning the ``HelixJobSpec`` to the factory::

    from nemo_helix_plugin.jobs.profile import stamp_profile

    def compile(self, *, profile, ...):
        spec = HelixJobSpec(steps=[...])
        stamp_profile(spec, profile or "default")
        return spec

The helper mutates *spec* in place and also returns it for chaining.

Why it lives in ``nemo_helix_plugin`` and not in the Jobs service:

Each step's ``executor`` is a plugin-owned discriminated union. All provider
models carry a ``profile: str`` field — that's the only attribute the stamper
touches. Keeping the helper in ``nemo_helix_plugin``
alongside the factory avoids dragging plugin-service code through the Jobs
service's internals and matches where ``add_job_routes()`` already lives.
"""

from __future__ import annotations

from typing import Any


def stamp_profile(spec: Any, profile: str) -> Any:
    """Apply *profile* to every step executor that hasn't set one already.

    Per-step overrides set by the compiler win — this helper only fills in
    the blanks. An "unset" profile is either the attribute missing, the
    attribute set to ``None``, or the attribute set to the empty string.
    Everything else is treated as a deliberate per-step override and left
    alone.

    Args:
        spec: A ``HelixJobSpec`` (or duck-typed equivalent — anything
            with a ``steps`` attribute, where each step has an ``executor``
            attribute with a ``profile`` attribute).
        profile: The job-level profile name. Typically the submitter's
            ``--profile <p>`` value; callers that want to default explicitly
            should pass ``profile or "default"``.

    Returns:
        The same *spec* object, for call chaining.

    Raises:
        TypeError: If *spec* lacks a ``steps`` attribute or any step's
            ``executor`` lacks ``profile``.
    """
    if isinstance(spec, dict):
        steps = spec.get("steps")
    else:
        steps = getattr(spec, "steps", None)
    if steps is None:
        raise TypeError(f"stamp_profile: spec has no 'steps' attribute (got {type(spec).__name__})")

    for step in steps:
        executor = _resolve_executor(step)
        current = _get_profile(executor)
        if current in (None, ""):
            _set_profile(executor, profile)

    return spec


# ---------------------------------------------------------------------------
# Accessor helpers — tolerate both attribute-style and mapping-style step
# executors so the stamper works against Pydantic models, TypedDicts, and
# plain dicts alike (the SDK's generated types use TypedDict-style ``*Param``
# shapes that behave like dicts on the wire).
# ---------------------------------------------------------------------------


def _resolve_executor(step: Any) -> Any:
    if hasattr(step, "executor"):
        return step.executor
    if isinstance(step, dict) and "executor" in step:
        return step["executor"]
    raise TypeError(f"stamp_profile: step has no 'executor' field (got {type(step).__name__})")


def _get_profile(executor: Any) -> str | None:
    if isinstance(executor, dict):
        return executor.get("profile")
    return getattr(executor, "profile", None)


def _set_profile(executor: Any, profile: str) -> None:
    if isinstance(executor, dict):
        executor["profile"] = profile
    else:
        try:
            setattr(executor, "profile", profile)
        except AttributeError as exc:
            raise TypeError(
                f"stamp_profile: cannot set profile on executor (type={type(executor).__name__}): {exc}"
            ) from exc
