# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Fabric search-space parsing for optimizer phases."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from nemo_optimization.config_overlay import get_by_dotted_path

SUPPORTED_PARAM_TYPES = frozenset({"fabric"})
DEFAULT_PARAM_TYPE = "fabric"
LEGACY_SEARCH_SPACE_ERROR = "optimizer.optimizable_params is no longer supported; rename it to optimizer.search_space."


class SearchSpaceError(ValueError):
    """Raised when a search-space entry is invalid."""


@dataclass(frozen=True)
class NumericSearchSpaceSpec:
    """One non-prompt Fabric search-space dimension."""

    path: str
    param_type: str = DEFAULT_PARAM_TYPE
    values: tuple[Any, ...] | None = None
    low: int | float | None = None
    high: int | float | None = None
    log: bool = False
    step: int | float | None = None

    @classmethod
    def from_mapping(cls, name: str, spec: Mapping[str, Any]) -> NumericSearchSpaceSpec:
        param_type, path = _common_entry_fields(name, spec)
        values = spec.get("values")
        if values is not None:
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise SearchSpaceError("'values' must be a non-string sequence.")
            if not values:
                raise SearchSpaceError("'values' must not be empty.")
            if spec.get("low") is not None or spec.get("high") is not None:
                raise SearchSpaceError("'values' is mutually exclusive with 'low' and 'high'.")
            return cls(path=path, param_type=param_type, values=tuple(values))

        low = spec.get("low")
        high = spec.get("high")
        if (low is None) != (high is None):
            raise SearchSpaceError("Range search spaces require both 'low' and 'high'.")
        if low is None or high is None:
            raise SearchSpaceError("Search space entry must define either 'values' or both 'low' and 'high'.")
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, (int, float))
            or not isinstance(high, (int, float))
        ):
            raise SearchSpaceError(f"'low' and 'high' must be numbers; got low={low!r}, high={high!r}.")
        if not _is_finite_number(low) or not _is_finite_number(high):
            raise SearchSpaceError(f"'low' and 'high' must be finite numbers; got low={low!r}, high={high!r}.")
        if low >= high:
            raise SearchSpaceError(f"'low' must be less than 'high'; got low={low}, high={high}.")
        step = spec.get("step")
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, (int, float)):
                raise SearchSpaceError(f"'step' must be a number; got {step!r}.")
            if not _is_finite_number(step) or step <= 0:
                raise SearchSpaceError(f"'step' must be a positive finite number; got {step!r}.")
            if isinstance(low, int) and isinstance(high, int) and not isinstance(step, int):
                raise SearchSpaceError(f"'step' must be a positive integer for integer ranges; got {step!r}.")
        log = spec.get("log", False)
        if not isinstance(log, bool):
            raise SearchSpaceError("'log' must be a boolean.")
        if isinstance(low, int) and isinstance(high, int) and log and step is not None and step > 1:
            raise SearchSpaceError("Integer ranges with 'log: true' require 'step' to be 1.")

        return cls(
            path=path,
            param_type=param_type,
            low=low,
            high=high,
            log=log,
            step=step,
        )


@dataclass(frozen=True)
class PromptSearchSpaceSpec:
    """One prompt Fabric search-space dimension."""

    path: str
    purpose: str
    param_type: str = DEFAULT_PARAM_TYPE
    format: str | None = None
    initial_prompt: str | None = None
    required_variables: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, name: str, spec: Mapping[str, Any]) -> PromptSearchSpaceSpec:
        param_type, path = _common_entry_fields(name, spec)
        unsupported = sorted(key for key in ("values", "low", "high", "log", "step") if key in spec)
        if unsupported:
            raise SearchSpaceError(
                f"Prompt search-space entry {name!r} must not define numeric search keys: {unsupported}."
            )

        purpose = spec.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            raise SearchSpaceError(f"Prompt search-space entry {name!r} requires non-empty 'purpose'.")

        raw_format = spec.get("format")
        if raw_format is not None and (not isinstance(raw_format, str) or not raw_format.strip()):
            raise SearchSpaceError(f"Prompt search-space entry {name!r} has malformed 'format'.")

        raw_initial_prompt = spec.get("initial_prompt")
        if raw_initial_prompt is not None and not isinstance(raw_initial_prompt, str):
            raise SearchSpaceError(f"Prompt search-space entry {name!r} field 'initial_prompt' must be a string.")

        required_variables = _parse_required_variables(name, spec.get("required_variables", ()))

        return cls(
            path=path,
            purpose=purpose.strip(),
            param_type=param_type,
            format=raw_format.strip() if isinstance(raw_format, str) else None,
            initial_prompt=raw_initial_prompt,
            required_variables=required_variables,
        )


SearchSpaceSpec = NumericSearchSpaceSpec | PromptSearchSpaceSpec


def parse_all_search_space(optimizer: Mapping[str, Any]) -> dict[str, SearchSpaceSpec]:
    """Parse ``optimizer.search_space``."""
    if "optimizable_params" in optimizer:
        raise SearchSpaceError(LEGACY_SEARCH_SPACE_ERROR)
    raw = optimizer.get("search_space")
    if not isinstance(raw, Mapping):
        raise SearchSpaceError("optimizer.search_space must be a mapping of param names to typed specs.")

    space: dict[str, SearchSpaceSpec] = {}
    for name, spec in raw.items():
        if not isinstance(name, str):
            raise SearchSpaceError("Search-space keys must be strings (logical param names).")
        if not isinstance(spec, Mapping):
            raise SearchSpaceError(f"Search space entry {name!r} must be a mapping.")
        space[name] = parse_search_space_entry(name, spec)
    if not space:
        raise SearchSpaceError("optimizer.search_space must declare at least one dimension.")
    return space


def parse_search_space_entry(name: str, spec: Mapping[str, Any]) -> SearchSpaceSpec:
    """Parse one shared search-space entry."""

    if _is_prompt_dimension(name, spec):
        return PromptSearchSpaceSpec.from_mapping(name, spec)
    return NumericSearchSpaceSpec.from_mapping(name, spec)


def parse_numeric_search_space(optimizer: Mapping[str, Any]) -> dict[str, NumericSearchSpaceSpec]:
    """Parse numeric dimensions from ``optimizer.search_space``."""

    space = {
        name: spec
        for name, spec in parse_all_search_space(optimizer).items()
        if isinstance(spec, NumericSearchSpaceSpec)
    }
    if not space:
        raise SearchSpaceError("optimizer.search_space must declare at least one non-prompt dimension.")
    return space


def parse_prompt_search_space(
    optimizer: Mapping[str, Any],
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, PromptSearchSpaceSpec]:
    """Parse prompt dimensions from ``optimizer.search_space``."""

    space = {
        name: spec
        for name, spec in parse_all_search_space(optimizer).items()
        if isinstance(spec, PromptSearchSpaceSpec)
    }
    if not space:
        raise SearchSpaceError("optimizer.search_space must declare at least one prompt dimension.")
    if payload is not None:
        validate_prompt_search_space_payload(payload, space)
        space = _with_inferred_required_variables(payload, space)
    return space


def validate_prompt_search_space_payload(
    payload: Mapping[str, Any],
    search_space: Mapping[str, PromptSearchSpaceSpec],
) -> None:
    """Ensure each prompt path resolves to the initial prompt string."""

    for name, spec in search_space.items():
        initial_prompt = initial_prompt_for_spec(payload, name=name, spec=spec)
        validate_prompt_variables(
            prompt_name=name,
            prompt=initial_prompt,
            required_variables=spec.required_variables,
            context="initial prompt",
        )


def initial_prompt_for_spec(payload: Mapping[str, Any], *, name: str, spec: PromptSearchSpaceSpec) -> str:
    """Return the explicit prompt seed or resolve it from the Fabric payload."""

    if spec.initial_prompt is not None:
        return spec.initial_prompt
    try:
        initial_prompt = get_by_dotted_path(payload, spec.path)
    except KeyError as exc:
        raise SearchSpaceError(
            f"Prompt search-space entry {name!r} path {spec.path!r} does not resolve in the Fabric payload."
        ) from exc
    if not isinstance(initial_prompt, str):
        raise SearchSpaceError(
            f"Prompt search-space entry {name!r} path {spec.path!r} must resolve to a string "
            f"(got {type(initial_prompt).__name__})."
        )
    return initial_prompt


def required_prompt_variables(spec: PromptSearchSpaceSpec, initial_prompt: str) -> tuple[str, ...]:
    """Return explicit required variables or infer them from the initial prompt."""

    return spec.required_variables or extract_prompt_variables(initial_prompt)


def _with_inferred_required_variables(
    payload: Mapping[str, Any],
    search_space: Mapping[str, PromptSearchSpaceSpec],
) -> dict[str, PromptSearchSpaceSpec]:
    hydrated: dict[str, PromptSearchSpaceSpec] = {}
    for name, spec in search_space.items():
        initial_prompt = initial_prompt_for_spec(payload, name=name, spec=spec)
        variables = required_prompt_variables(spec, initial_prompt)
        hydrated[name] = replace(spec, required_variables=variables)
    return hydrated


def validate_prompt_variables(
    *,
    prompt_name: str,
    prompt: str,
    required_variables: Sequence[str],
    context: str,
) -> None:
    missing = missing_prompt_variables(prompt, required_variables)
    if missing:
        raise SearchSpaceError(
            f"Prompt {prompt_name!r} {context} is missing required variable(s): {', '.join(missing)}."
        )


def missing_prompt_variables(prompt: str, required_variables: Sequence[str]) -> tuple[str, ...]:
    if not required_variables:
        return ()
    present = set(extract_prompt_variables(prompt))
    return tuple(variable for variable in required_variables if variable not in present)


def extract_prompt_variables(prompt: str) -> tuple[str, ...]:
    """Return variable-like placeholders from common prompt template syntaxes."""

    variables: set[str] = set()
    variables.update(match.group(1) for match in _JINJA_VARIABLE_RE.finditer(prompt))
    variables.update(match.group(1) for match in _FORMAT_VARIABLE_RE.finditer(prompt))
    return tuple(sorted(variables))


def suggestions_by_path(
    search_space: Mapping[str, SearchSpaceSpec],
    suggestions: Mapping[str, Any],
) -> dict[str, Any]:
    """Map logical suggestions onto applicator ``path`` keys."""
    by_path: dict[str, Any] = {}
    for name, value in suggestions.items():
        spec = search_space.get(name)
        if spec is None:
            raise SearchSpaceError(f"Suggestion {name!r} is not in the parsed search space.")
        if spec.path in by_path:
            raise SearchSpaceError(
                f"Search-space paths collide at {spec.path!r} "
                f"(params {[n for n, s in search_space.items() if s.path == spec.path]})."
            )
        by_path[spec.path] = value
    return by_path


def _is_prompt_dimension(name: str, spec: Mapping[str, Any]) -> bool:
    raw = spec.get("is_prompt", False)
    if not isinstance(raw, bool):
        raise SearchSpaceError(f"Search space entry {name!r} field 'is_prompt' must be a boolean.")
    return raw


def _common_entry_fields(name: str, spec: Mapping[str, Any]) -> tuple[str, str]:
    param_type = spec.get("type", DEFAULT_PARAM_TYPE)
    if not isinstance(param_type, str):
        raise SearchSpaceError(f"Search space entry {name!r} field 'type' must be a string.")
    param_type = param_type.strip().lower()
    if param_type not in SUPPORTED_PARAM_TYPES:
        supported = ", ".join(sorted(SUPPORTED_PARAM_TYPES))
        raise SearchSpaceError(
            f"Search space entry {name!r} has unsupported type {param_type!r}; supported types: {supported}."
        )

    path = spec.get("path")
    if not isinstance(path, str) or not path.strip():
        raise SearchSpaceError(f"Search space entry {name!r} requires 'path' (Fabric overlay dotted path).")
    path = path.strip()
    if any(not segment for segment in path.split(".")):
        raise SearchSpaceError(f"Search space entry {name!r} path {path!r} must be a valid dotted path.")
    return param_type, path


def _parse_required_variables(name: str, raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SearchSpaceError(f"Prompt search-space entry {name!r} field 'required_variables' must be a list.")
    variables: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise SearchSpaceError(
                f"Prompt search-space entry {name!r} field 'required_variables' must contain non-empty strings."
            )
        variable = item.strip()
        if variable not in seen:
            variables.append(variable)
            seen.add(variable)
    return tuple(variables)


def _is_finite_number(value: int | float) -> bool:
    return math.isfinite(float(value))


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_JINJA_VARIABLE_RE = re.compile(r"{{\s*(" + _IDENTIFIER + r")\s*}}")
_FORMAT_VARIABLE_RE = re.compile(r"(?<!{){(" + _IDENTIFIER + r")(?:![^}:]+)?(?::[^}]*)?}(?!})")
