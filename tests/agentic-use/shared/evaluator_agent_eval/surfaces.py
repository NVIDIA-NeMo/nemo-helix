# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surface detection for Evaluator agent benchmark artifacts."""

import re
from collections.abc import Sequence
from pathlib import Path

from evaluator_agent_eval.schemas import SurfaceName
from pydantic import BaseModel, ConfigDict, Field


class SurfaceEvidence(BaseModel):
    """Evidence used to infer which product surfaces an agent touched."""

    model_config = ConfigDict(extra="forbid")

    final_answer_text: str = ""
    raw_logs: list[str] = Field(default_factory=list)
    command_argvs: list[list[str]] = Field(default_factory=list)
    changed_paths: list[str] = Field(default_factory=list)


class SurfaceDetectionResult(BaseModel):
    """Detected surfaces and concrete forbidden evidence."""

    model_config = ConfigDict(extra="forbid")

    observed_surfaces: list[SurfaceName]
    forbidden_surface_hits: list[str]


def detect_surfaces(
    evidence: SurfaceEvidence,
    *,
    forbidden_patterns: Sequence[str] = (),
) -> SurfaceDetectionResult:
    """Infer surface usage from final answer, logs, commands, and paths."""
    soft_text = "\n".join([evidence.final_answer_text, *evidence.raw_logs]).lower()
    hard_text = "\n".join([" ".join(argv) for argv in evidence.command_argvs] + list(evidence.changed_paths)).lower()
    haystack = "\n".join(part for part in [soft_text, hard_text] if part)

    observed: list[SurfaceName] = []
    if "packages/nemo_evaluator_sdk" in haystack or "nemo_evaluator_sdk" in haystack:
        observed.append("standalone_sdk")
    if _command_contains(evidence.command_argvs, ["nemo", "evaluation"]) or contains_nonnegated_substring(
        soft_text, "nemo evaluation"
    ):
        observed.append("cli")
    if (
        contains_nonnegated_substring(soft_text, "plugin sdk")
        or contains_nonnegated_substring(soft_text, "plugin_sdk")
        or contains_nonnegated_substring(soft_text, "nemo-helix-plugin")
        or contains_nonnegated_substring(soft_text, "nemo-plugin")
    ):
        observed.append("plugin_sdk")
    if _contains_legacy_service_path(hard_text) or contains_nonnegated_legacy_service_path(soft_text):
        observed.append("legacy_service")

    observed_surfaces: list[SurfaceName] = _dedupe(observed)
    if not observed_surfaces:
        observed_surfaces = ["unknown"]
    hits = _forbidden_hits(soft_text=soft_text, hard_text=hard_text, forbidden_patterns=forbidden_patterns)
    return SurfaceDetectionResult(
        observed_surfaces=observed_surfaces,
        forbidden_surface_hits=hits,
    )


def evidence_from_artifacts(
    *,
    final_answer_text: str,
    raw_text: str = "",
    command_json_path: str | Path | None = None,
) -> SurfaceEvidence:
    """Build surface evidence from common runner artifacts."""
    command_argvs: list[list[str]] = []
    if command_json_path is not None:
        command_argvs = _read_command_argvs(Path(command_json_path))
    return SurfaceEvidence(
        final_answer_text=final_answer_text, raw_logs=[raw_text] if raw_text else [], command_argvs=command_argvs
    )


def _forbidden_hits(*, soft_text: str, hard_text: str, forbidden_patterns: Sequence[str]) -> list[str]:
    patterns = [pattern for pattern in forbidden_patterns if pattern]
    hits: list[str] = []
    for pattern in patterns:
        lowered = pattern.lower()
        if lowered in hard_text or contains_nonnegated_substring(soft_text, lowered):
            hits.append(pattern)
    return hits


def _contains_legacy_service_path(haystack: str) -> bool:
    return re.search(r"services/[a-z0-9_.-]|services\\(?![nrt])[a-z0-9_.-]", haystack) is not None


def contains_nonnegated_legacy_service_path(text: str) -> bool:
    pattern = re.compile(r"services/[a-z0-9_.*-]+|services\\(?![nrt])[a-z0-9_.*-]+")
    return _contains_nonnegated_match(text, pattern)


def contains_nonnegated_substring(text: str, pattern: str) -> bool:
    if not text or not pattern:
        return False
    return _contains_nonnegated_match(text, re.compile(re.escape(pattern)))


def _read_command_argvs(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    argv = payload.get("argv") if isinstance(payload, dict) else None
    if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
        return []
    return [argv]


def _command_contains(command_argvs: Sequence[Sequence[str]], tokens: Sequence[str]) -> bool:
    token_set = {token.lower() for token in tokens}
    for argv in command_argvs:
        if token_set.issubset({part.lower() for part in argv}):
            return True
    return False


def _dedupe(surfaces: Sequence[SurfaceName]) -> list[SurfaceName]:
    seen: set[SurfaceName] = set()
    result: list[SurfaceName] = []
    for surface in surfaces:
        if surface in seen:
            continue
        seen.add(surface)
        result.append(surface)
    return result


def _contains_nonnegated_match(text: str, pattern: re.Pattern[str]) -> bool:
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 80) : match.start()]
        if _prefix_contains_negation(prefix):
            continue
        return True
    return False


def _prefix_contains_negation(prefix: str) -> bool:
    return (
        re.search(
            r"\b(?:do not|don't|does not|doesn't|did not|didn't|without|avoid|avoids|avoiding|instead of|rather than|not use|never use|no)\b",
            prefix,
        )
        is not None
    )
