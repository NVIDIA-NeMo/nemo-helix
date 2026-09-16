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


def test_a_leaf_that_is_not_a_model_field_lands_in_settings() -> None:
    result = apply_tuned_params_to_spec(_spec(), {"models.default.top_p": 0.9})
    assert result["harnesses"]["hermes"]["model"]["settings"]["top_p"] == 0.9
    assert "top_p" not in result["harnesses"]["hermes"]["model"]


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
