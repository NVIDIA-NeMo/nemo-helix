# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply tuned study parameters onto a ``nemo-agents-spec-v1`` agent config.

**Search-space paths are Fabric-shaped.**  The study mutates the Fabric package
the harness actually executes (``models.default.temperature``,
``harness.settings.max_tokens``), and this module translates the winning values
back onto the platform spec the optimized agent is persisted as.  The two shapes
differ — most visibly, Fabric has one ``harness`` block while the spec has a
``harnesses`` map keyed by name — so every translation lives here, in one place.

Two failure modes this module exists to prevent, both of which produce an agent
that looks optimized and is not:

1. *The overlay invents a key.*  The Fabric ``models.default`` block is
   synthesized by the translator and frequently has no counterpart in the stored
   config, so the overlay writes only to locations it can name on the spec and
   raises when a tuned parameter matches nothing.
2. *The tuned value never reaches the harness.*  A ``models.default.<leaf>``
   parameter is only written when ``<leaf>`` is a field the spec->fabric
   translator round-trips (a declared ``ModelConfig`` field).  A leaf outside
   that set — e.g. ``top_p`` — has no home in the translator's output: writing it
   into the harness model's ``settings`` looks like it worked, but re-translating
   the spec back to a Fabric package does not carry ``settings`` into the adapter
   fields harnesses read (the hermes adapter reads ``top_p`` from ``extensions``,
   not ``settings``), so the tuned value silently never reaches the agent.

:func:`search_space_path_problem` is the single contract check: it answers both
"can the study actually tune this path?" and "can the overlay write the result
back?" for one path, so the two halves cannot disagree silently.  The ``nat``
job calls it for every declared search-space path *before* the study runs.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: Declared fields of ``nemo_agents_plugin.agent_config.ModelConfig`` that the
#: spec->fabric translator round-trips.  A tuned leaf outside this set is refused
#: rather than written to ``settings`` — see the module docstring for why.
_MODEL_FIELDS = frozenset({"provider", "model", "api_key_env", "base_url", "temperature"})

#: Fabric package prefixes this module knows how to translate onto the spec.
_FABRIC_DEFAULT_MODEL_PREFIX = "models.default."
_FABRIC_HARNESS_SETTINGS_PREFIX = "harness.settings."
_FABRIC_HARNESS_KEY = "harness"

#: The spec-side key that is *not* a Fabric key.  A search space that names it is
#: the inverted-contract mistake this module refuses by name rather than by
#: accident: it applies cleanly to the spec and does nothing at all in the study.
_SPEC_HARNESSES_KEY = "harnesses"

_FABRIC_SHAPED_EXAMPLES = "'models.default.temperature', 'harness.settings.max_tokens'"

_MATCHED_NOTHING = (
    "matched nothing in the agent spec. Search-space paths must address a value the stored agent "
    "actually has — for a model parameter, the selected harness's model block."
)


class SpecOverlayError(RuntimeError):
    """Raised when a tuned parameter cannot be applied to the agent spec."""


def apply_tuned_params_to_spec(spec_config: dict[str, Any], by_path: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *spec_config* with each tuned path in *by_path* applied."""
    optimized = copy.deepcopy(spec_config)
    for dotted_path, value in by_path.items():
        targets, problem = _resolve_targets(optimized, dotted_path)
        if problem is not None:
            raise SpecOverlayError(f"Tuned parameter {dotted_path!r} cannot be applied: {problem}")
        if not targets:
            raise SpecOverlayError(f"Tuned parameter {dotted_path!r} {_MATCHED_NOTHING}")
        for target, leaf in targets:
            target[leaf] = value
        logger.info("Applied tuned parameter %s=%r to the agent spec", dotted_path, value)
    return optimized


def search_space_path_problem(
    dotted_path: str,
    *,
    fabric_payload: Mapping[str, Any],
    spec_config: Mapping[str, Any],
) -> str | None:
    """Why *dotted_path* cannot be tuned end to end, or ``None`` when it can.

    Checks both halves of the contract against the run's own inputs:

    * the study must genuinely mutate the agent — ``config_overlay.set_by_dotted_path``
      *creates* missing keys, so a path that addresses nothing in *fabric_payload*
      leaves every trial running the agent's original values while the trial
      config grows a dead key;
    * the overlay must be able to write the winning value back onto *spec_config*.

    A path that resolves in the Fabric payload is tunable: the value is already
    there, which is the only evidence available that the harness reads it.  The
    one exception is a declared ``ModelConfig`` field under ``models.default``,
    where the translator's schema is the evidence — the adapter reads the field
    whether or not the agent currently sets it.
    """
    targets, problem = _resolve_targets(copy.deepcopy(dict(spec_config)), dotted_path)
    if problem is not None:
        return problem
    if not targets:
        return _MATCHED_NOTHING

    if _resolves(fabric_payload, dotted_path) or _is_settable_default_model_field(fabric_payload, dotted_path):
        return None
    return (
        "it does not resolve against the Fabric package the study mutates, so every trial would "
        "write a dead key and run with the agent's unchanged value. Declare the value on the agent "
        "first (with the baseline you want to tune away from), or use a Fabric-shaped path that "
        f"exists ({_FABRIC_SHAPED_EXAMPLES})"
    )


def _resolves(config: Mapping[str, Any], dotted_path: str) -> bool:
    """Whether every segment of *dotted_path* already exists in *config*."""
    cursor: Any = config
    for key in dotted_path.split("."):
        if not isinstance(cursor, Mapping) or key not in cursor:
            return False
        cursor = cursor[key]
    return True


def _is_settable_default_model_field(fabric_payload: Mapping[str, Any], dotted_path: str) -> bool:
    """Whether *dotted_path* names a declared model field the study may create.

    ``models.default`` is typed by the spec->fabric translator, so a declared
    field reaches the adapter even when the agent leaves it unset today.
    """
    if not dotted_path.startswith(_FABRIC_DEFAULT_MODEL_PREFIX):
        return False
    leaf = dotted_path[len(_FABRIC_DEFAULT_MODEL_PREFIX) :]
    return leaf in _MODEL_FIELDS and _resolves(fabric_payload, "models.default")


def _resolve_targets(
    spec_config: dict[str, Any],
    dotted_path: str,
) -> tuple[list[tuple[dict[str, Any], str]], str | None]:
    """Resolve a Fabric-shaped path to ``(container, leaf)`` write targets on the spec.

    Returns ``(targets, problem)``.  An empty ``targets`` with no ``problem``
    means the path named a location the stored spec does not have.
    """
    if dotted_path.startswith(_FABRIC_DEFAULT_MODEL_PREFIX):
        leaf = dotted_path[len(_FABRIC_DEFAULT_MODEL_PREFIX) :]
        if leaf not in _MODEL_FIELDS:
            return [], (
                f"{leaf!r} is not one of the fields the spec->fabric translator round-trips "
                f"({sorted(_MODEL_FIELDS)}). Writing it into the harness model's 'settings' would "
                "look like it worked, but re-translating the spec back to a Fabric package would "
                "silently drop it before it ever reaches the running agent (e.g. the hermes adapter "
                "reads 'top_p' from 'extensions', not 'settings'). Tune a harness setting the agent "
                "already declares instead, e.g. 'harness.settings.<field>'"
            )
        return _model_block_targets(spec_config, leaf), None

    if dotted_path.startswith(_FABRIC_HARNESS_SETTINGS_PREFIX):
        rest = dotted_path[len(_FABRIC_HARNESS_SETTINGS_PREFIX) :].split(".")
        default_harness = spec_config.get("default_harness")
        if not isinstance(default_harness, str):
            return [], (
                "the agent spec declares no 'default_harness', so there is no harness whose "
                "'settings' the Fabric 'harness.settings' block corresponds to"
            )
        return _existing_target(spec_config, [_SPEC_HARNESSES_KEY, default_harness, "settings", *rest]), None

    if dotted_path == _FABRIC_HARNESS_KEY or dotted_path.startswith(f"{_FABRIC_HARNESS_KEY}."):
        return [], (
            "only 'harness.settings.<field>' round-trips between the Fabric package and the agent "
            "spec. The rest of the Fabric harness block (adapter_id, resolution) is derived from "
            "the spec's harness 'kind' and cannot be written back"
        )

    if dotted_path.split(".", 1)[0] == _SPEC_HARNESSES_KEY:
        return [], (
            f"{_SPEC_HARNESSES_KEY!r} is a spec key, not a Fabric key. Search-space paths are "
            f"Fabric-shaped, and the Fabric harness block is {_FABRIC_HARNESS_KEY!r} (singular): "
            "the study would tune nothing. Use 'harness.settings.<field>' instead"
        )

    return _existing_target(spec_config, dotted_path.split(".")), None


def _model_block_targets(spec_config: dict[str, Any], leaf: str) -> list[tuple[dict[str, Any], str]]:
    """Every model block the spec actually has that *leaf* must be written to.

    Mirrors switchyard's ``_rewrite_model``: the default harness's model is the
    real target, and an explicit ``models.default`` is updated too when present
    so the two cannot drift apart. Callers only reach this once *leaf* has been
    verified to be a declared, round-trippable ``ModelConfig`` field — never
    written blind, and never falls back to inventing a ``settings`` entry.
    """
    targets: list[tuple[dict[str, Any], str]] = []

    harnesses = spec_config.get(_SPEC_HARNESSES_KEY)
    default_harness = spec_config.get("default_harness")
    if isinstance(harnesses, dict) and isinstance(harnesses.get(default_harness), dict):
        model = harnesses[default_harness].get("model")
        if isinstance(model, dict):
            targets.append((model, leaf))

    models = spec_config.get("models")
    if isinstance(models, dict) and isinstance(models.get("default"), dict):
        targets.append((models["default"], leaf))

    return targets


def _existing_target(config: dict[str, Any], keys: list[str]) -> list[tuple[dict[str, Any], str]]:
    """The write target for a dotted path, only when every segment already exists."""
    cursor: Any = config
    for key in keys[:-1]:
        if not isinstance(cursor, dict) or not isinstance(cursor.get(key), dict):
            return []
        cursor = cursor[key]
    if not isinstance(cursor, dict) or keys[-1] not in cursor:
        return []
    return [(cursor, keys[-1])]
