# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_optimization.spec_overlay import (
    SpecOverlayError,
    apply_tuned_params_to_spec,
    search_space_path_problem,
)


def _spec() -> dict:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "my-agent",
        "default_harness": "hermes",
        "harnesses": {
            "hermes": {
                "kind": "hermes",
                "model": {"provider": "openai", "model": "m", "temperature": 0.0},
                "settings": {"max_tokens": 256},
            }
        },
        "models": {"judge": {"provider": "openai", "model": "j"}},
    }


def test_a_fabric_model_path_lands_on_the_default_harness_model() -> None:
    result = apply_tuned_params_to_spec(_spec(), {"models.default.temperature": 0.7})
    assert result["harnesses"]["hermes"]["model"]["temperature"] == 0.7
    # The spec never had a models.default block; applying the overlay must not invent one —
    # a key-creating setter here is exactly the silent-failure shape this overlay exists to
    # prevent (see spec_overlay.py's module docstring).
    assert "default" not in result["models"]


def test_a_leaf_that_is_not_a_declared_model_field_is_refused() -> None:
    """top_p does not round-trip through the spec->fabric translator (it lives in the fabric
    model's 'extensions', not 'settings'), so writing it into settings would silently never
    reach the running agent. The overlay must refuse it, not silently no-op it into settings.
    """
    with pytest.raises(SpecOverlayError, match="top_p"):
        apply_tuned_params_to_spec(_spec(), {"models.default.top_p": 0.9})


def test_a_dotted_leaf_under_models_default_is_refused_not_invented() -> None:
    """A leaf with dots of its own (e.g. from a search-space path like
    'models.default.settings.top_p') must not create a literal 'settings.top_p' key under
    settings — it is refused for the same round-trip reason as a plain non-model leaf.
    """
    spec = _spec()
    spec["harnesses"]["hermes"]["model"]["settings"] = {"top_p": 1.0}
    with pytest.raises(SpecOverlayError, match="settings.top_p"):
        apply_tuned_params_to_spec(spec, {"models.default.settings.top_p": 0.3})
    # Confirm nothing was invented: the original settings dict is untouched by the attempt.
    assert spec["harnesses"]["hermes"]["model"]["settings"] == {"top_p": 1.0}


def test_an_existing_models_default_is_also_updated() -> None:
    spec = _spec()
    spec["models"]["default"] = {"provider": "openai", "model": "m", "temperature": 0.0}
    result = apply_tuned_params_to_spec(spec, {"models.default.temperature": 0.7})
    assert result["models"]["default"]["temperature"] == 0.7
    assert result["harnesses"]["hermes"]["model"]["temperature"] == 0.7


def test_a_fabric_harness_settings_path_lands_on_the_default_harness() -> None:
    """The Fabric package has one 'harness'; the spec has a 'harnesses' map. The overlay
    owns that translation, so the path the study can really apply is also the one the
    overlay accepts.
    """
    result = apply_tuned_params_to_spec(_spec(), {"harness.settings.max_tokens": 512})
    assert result["harnesses"]["hermes"]["settings"]["max_tokens"] == 512


def test_a_spec_shaped_harnesses_path_is_refused() -> None:
    """Replaces the former ``test_a_spec_shaped_path_is_applied_verbatim``.

    ``harnesses.<name>.settings.<field>`` applies cleanly to the spec and does *nothing*
    in the study: ``set_by_dotted_path`` creates a dead 'harnesses' key in the Fabric
    payload, so the trial runs the agent's original settings. Accepting it here is what
    let a never-measured value reach a registered agent.
    """
    with pytest.raises(SpecOverlayError, match="not a Fabric key"):
        apply_tuned_params_to_spec(_spec(), {"harnesses.hermes.settings.max_tokens": 512})


def test_a_non_settings_harness_path_is_refused() -> None:
    with pytest.raises(SpecOverlayError, match="harness.settings"):
        apply_tuned_params_to_spec(_spec(), {"harness.adapter_id": "nvidia.fabric.claude"})


def test_a_path_that_matches_nothing_fails_loudly() -> None:
    with pytest.raises(SpecOverlayError, match="matched nothing"):
        apply_tuned_params_to_spec(_spec(), {"llms.nowhere.temperature": 0.5})


def test_the_input_is_not_mutated() -> None:
    spec = _spec()
    apply_tuned_params_to_spec(spec, {"models.default.temperature": 0.7})
    assert spec["harnesses"]["hermes"]["model"]["temperature"] == 0.0


def _fabric_payload() -> dict:
    """The shape ``to_fabric_agent_package`` produces for :func:`_spec` (harness singular)."""
    return {
        "schema_version": "fabric.agent/v1alpha1",
        "harness": {"adapter_id": "nvidia.fabric.hermes", "settings": {"max_tokens": 256}},
        "models": {
            "default": {"provider": "openai", "model": "m", "temperature": 0.0},
            "judge": {"provider": "openai", "model": "j"},
        },
    }


def test_a_path_that_works_on_both_halves_has_no_problem() -> None:
    for path in ("models.default.temperature", "harness.settings.max_tokens"):
        assert search_space_path_problem(path, fabric_payload=_fabric_payload(), spec_config=_spec()) is None


def test_a_declared_model_field_the_agent_has_not_set_is_still_tunable() -> None:
    """The translator types ``models.default``, so a declared field reaches the adapter
    even when the agent leaves it unset today."""
    payload = _fabric_payload()
    del payload["models"]["default"]["temperature"]
    assert search_space_path_problem("models.default.temperature", fabric_payload=payload, spec_config=_spec()) is None


def test_a_spec_shaped_path_the_study_cannot_apply_is_reported() -> None:
    problem = search_space_path_problem(
        "harnesses.hermes.settings.max_tokens",
        fabric_payload=_fabric_payload(),
        spec_config=_spec(),
    )
    assert problem is not None
    assert "not a Fabric key" in problem


def test_a_free_form_setting_the_agent_never_declares_is_reported() -> None:
    """Nothing proves the harness reads an invented free-form setting, and the spec has
    no baseline to overwrite — so it cannot be tuned."""
    problem = search_space_path_problem(
        "harness.settings.invented",
        fabric_payload=_fabric_payload(),
        spec_config=_spec(),
    )
    assert problem is not None
    assert "matched nothing" in problem


def test_a_non_round_trippable_model_leaf_is_reported() -> None:
    payload = _fabric_payload()
    payload["models"]["default"]["top_p"] = 1.0
    problem = search_space_path_problem("models.default.top_p", fabric_payload=payload, spec_config=_spec())
    assert problem is not None
    assert "top_p" in problem
