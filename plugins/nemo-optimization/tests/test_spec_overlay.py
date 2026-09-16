# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_optimization.spec_overlay import SpecOverlayError, apply_tuned_params_to_spec


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


def test_a_spec_shaped_path_is_applied_verbatim() -> None:
    result = apply_tuned_params_to_spec(_spec(), {"harnesses.hermes.settings.max_tokens": 512})
    assert result["harnesses"]["hermes"]["settings"]["max_tokens"] == 512


def test_a_path_that_matches_nothing_fails_loudly() -> None:
    with pytest.raises(SpecOverlayError, match="matched nothing"):
        apply_tuned_params_to_spec(_spec(), {"llms.nowhere.temperature": 0.5})


def test_the_input_is_not_mutated() -> None:
    spec = _spec()
    apply_tuned_params_to_spec(spec, {"models.default.temperature": 0.7})
    assert spec["harnesses"]["hermes"]["model"]["temperature"] == 0.0
