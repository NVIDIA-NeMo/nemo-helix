# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_optimization.backends.ga.config import parse_ga_prompt_optimizer_config
from nemo_optimization.backends.optuna.search_space import (
    SearchSpaceSpec,
    grid_trial_count,
    parse_search_space,
)
from nemo_optimization.optimizer_config import OptimizerConfigError, parse_optimizer_config
from nemo_optimization.search_space import (
    PromptSearchSpaceSpec,
    SearchSpaceError,
    parse_all_search_space,
    parse_prompt_search_space,
    suggestions_by_path,
)


class _FakeTrial:
    def suggest_categorical(self, name: str, choices):  # noqa: ANN001
        return choices[0]

    def suggest_int(self, name, low, high, *, log=False, step=1):  # noqa: ANN001
        return low

    def suggest_float(self, name, low, high, *, log=False, step=None):  # noqa: ANN001
        return low


def test_categorical_suggest() -> None:
    spec = SearchSpaceSpec.from_mapping(
        "top_p", {"type": "fabric", "path": "models.default.top_p", "values": [0.7, 0.85, 1.0]}
    )
    assert spec.suggest(_FakeTrial(), "top_p") == 0.7
    assert spec.path == "models.default.top_p"


def test_float_range_suggest() -> None:
    spec = SearchSpaceSpec.from_mapping(
        "temperature",
        {"type": "fabric", "path": "models.default.temperature", "low": 0.0, "high": 0.8, "step": 0.2},
    )
    assert spec.suggest(_FakeTrial(), "temperature") == 0.0


@pytest.mark.parametrize("step", [0.5, 1.0])
def test_integer_range_rejects_non_integer_step(step: float) -> None:
    with pytest.raises(SearchSpaceError, match="positive integer"):
        SearchSpaceSpec.from_mapping(
            "max_tool_calls",
            {"path": "harness.settings.workflow.max_tool_calls", "low": 1, "high": 10, "step": step},
        )


def test_log_integer_range_rejects_step_greater_than_one() -> None:
    with pytest.raises(SearchSpaceError, match="require 'step' to be 1"):
        SearchSpaceSpec.from_mapping(
            "max_tool_calls",
            {"path": "harness.settings.workflow.max_tool_calls", "low": 1, "high": 10, "step": 2, "log": True},
        )


def test_grid_values_from_explicit_values() -> None:
    spec = SearchSpaceSpec.from_mapping("top_p", {"path": "models.default.top_p", "values": [0.7, 0.85, 1.0]})
    assert spec.to_grid_values() == [0.7, 0.85, 1.0]


def test_grid_values_from_int_range() -> None:
    spec = SearchSpaceSpec.from_mapping(
        "max_tool_calls", {"path": "harness.settings.workflow.max_tool_calls", "low": 0, "high": 10, "step": 2}
    )
    assert spec.to_grid_values() == [0, 2, 4, 6, 8, 10]


def test_grid_values_from_float_range_includes_high() -> None:
    spec = SearchSpaceSpec.from_mapping(
        "temperature", {"path": "models.default.temperature", "low": 0.0, "high": 0.8, "step": 0.2}
    )
    assert spec.to_grid_values() == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_grid_requires_step_for_range() -> None:
    spec = SearchSpaceSpec.from_mapping("temperature", {"path": "models.default.temperature", "low": 0.0, "high": 0.8})
    with pytest.raises(SearchSpaceError, match="requires 'step'"):
        spec.to_grid_values()


def test_prompt_search_space_retains_prompt_metadata() -> None:
    spec = PromptSearchSpaceSpec.from_mapping(
        "system_prompt",
        {
            "type": "fabric",
            "path": "instructions.system.content",
            "is_prompt": True,
            "purpose": "Answer accurately.",
            "format": "text",
            "initial_prompt": "Answer {question}.",
            "required_variables": ["question"],
        },
    )

    assert spec.path == "instructions.system.content"
    assert spec.purpose == "Answer accurately."
    assert spec.format == "text"
    assert spec.initial_prompt == "Answer {question}."
    assert spec.required_variables == ("question",)


def test_parse_search_space_filters_prompt_entries_for_optuna() -> None:
    optimizer = {
        "search_space": {
            "temperature": {"path": "models.default.temperature", "values": [0.0, 0.2]},
            "system_prompt": {
                "path": "instructions.system.content",
                "is_prompt": True,
                "purpose": "Answer accurately.",
            },
        }
    }

    assert set(parse_all_search_space(optimizer)) == {"temperature", "system_prompt"}
    assert set(parse_search_space(optimizer)) == {"temperature"}
    assert set(parse_prompt_search_space(optimizer)) == {"system_prompt"}


def test_parse_search_space_rejects_legacy_optimizable_params() -> None:
    with pytest.raises(SearchSpaceError, match="rename it to optimizer.search_space"):
        parse_all_search_space(
            {"optimizable_params": {"temperature": {"path": "models.default.temperature", "values": [0.0, 0.2]}}}
        )


def test_parse_optimizer_config_rejects_legacy_optimizable_params() -> None:
    with pytest.raises(OptimizerConfigError, match="rename it to optimizer.search_space"):
        parse_optimizer_config({"optimizer": {"optimizable_params": {}}})


def test_parse_search_space_rejects_when_no_numeric_dimensions() -> None:
    with pytest.raises(SearchSpaceError, match="non-prompt"):
        parse_search_space(
            {
                "search_space": {
                    "prompt": {
                        "path": "instructions.system.content",
                        "is_prompt": True,
                        "purpose": "Answer accurately.",
                    }
                }
            }
        )


def test_optuna_search_space_spec_rejects_prompt_entry() -> None:
    with pytest.raises(SearchSpaceError, match="prompt-only"):
        SearchSpaceSpec.from_mapping(
            "prompt",
            {
                "path": "instructions.system.content",
                "is_prompt": True,
                "purpose": "Answer accurately.",
            },
        )


def test_parse_search_space_requires_path() -> None:
    with pytest.raises(SearchSpaceError, match="requires 'path'"):
        parse_search_space({"search_space": {"temperature": {"values": [0.0]}}})


def test_prompt_search_space_requires_purpose() -> None:
    with pytest.raises(SearchSpaceError, match="requires non-empty 'purpose'"):
        parse_prompt_search_space(
            {
                "search_space": {
                    "system_prompt": {
                        "path": "instructions.system.content",
                        "is_prompt": True,
                    }
                }
            }
        )


def test_prompt_search_space_requires_boolean_is_prompt() -> None:
    with pytest.raises(SearchSpaceError, match="must be a boolean"):
        parse_prompt_search_space(
            {
                "search_space": {
                    "system_prompt": {
                        "path": "instructions.system.content",
                        "is_prompt": "false",
                        "purpose": "Answer accurately.",
                    }
                }
            }
        )


def test_prompt_optimizer_config_requires_model_reference() -> None:
    payload = {
        "optimizer": {
            "prompt": {"enabled": True},
            "search_space": {
                "system_prompt": {
                    "path": "instructions.system.content",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                }
            },
        },
        "instructions": {"system": {"content": "Base prompt."}},
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
    }

    with pytest.raises(ValueError, match="prompt.model"):
        parse_ga_prompt_optimizer_config(payload)


def test_prompt_optimizer_config_rejects_non_string_backend() -> None:
    payload = {
        "optimizer": {
            "prompt": {"enabled": True, "backend": 123, "model": "prompt_optimizer"},
            "search_space": {
                "system_prompt": {
                    "path": "instructions.system.content",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                }
            },
        },
        "instructions": {"system": {"content": "Base prompt."}},
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
    }

    with pytest.raises(ValueError, match="prompt.backend"):
        parse_ga_prompt_optimizer_config(payload)


def test_prompt_optimizer_config_validates_path_resolves_to_string() -> None:
    payload = {
        "optimizer": {
            "prompt": {"enabled": True, "model": "prompt_optimizer"},
            "eval_metrics": {"score": {"direction": "maximize"}},
            "search_space": {
                "system_prompt": {
                    "path": "instructions.system.content",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                }
            },
        },
        "instructions": {"system": {"content": "Base prompt."}},
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
    }

    config = parse_ga_prompt_optimizer_config(payload)

    assert config.backend == "ga"
    assert config.model == "prompt_optimizer"
    assert config.search_space["system_prompt"].path == "instructions.system.content"


def test_prompt_optimizer_config_infers_required_variables_from_initial_prompt() -> None:
    payload = {
        "optimizer": {
            "prompt": {"enabled": True, "model": "prompt_optimizer"},
            "eval_metrics": {"score": {"direction": "maximize"}},
            "search_space": {
                "system_prompt": {
                    "path": "instructions.system.content",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                }
            },
        },
        "instructions": {"system": {"content": "Answer {{ user }} about {topic}."}},
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
    }

    config = parse_ga_prompt_optimizer_config(payload)

    assert config.search_space["system_prompt"].required_variables == ("topic", "user")


def test_prompt_optimizer_config_validates_explicit_initial_prompt_variables() -> None:
    payload = {
        "optimizer": {
            "prompt": {"enabled": True, "model": "prompt_optimizer"},
            "search_space": {
                "system_prompt": {
                    "path": "instructions.system.content",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                    "initial_prompt": "Answer plainly.",
                    "required_variables": ["topic"],
                }
            },
        },
        "instructions": {"system": {"content": "Payload prompt {topic}."}},
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
    }

    with pytest.raises(ValueError, match="missing required variable"):
        parse_ga_prompt_optimizer_config(payload)


def test_prompt_optimizer_config_rejects_non_string_prompt_path() -> None:
    payload = {
        "optimizer": {
            "prompt": {"enabled": True, "model": "prompt_optimizer"},
            "search_space": {
                "system_prompt": {
                    "path": "instructions.system",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                }
            },
        },
        "instructions": {"system": {"content": "Base prompt."}},
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
    }

    with pytest.raises(ValueError, match="must resolve to a string"):
        parse_ga_prompt_optimizer_config(payload)


def test_parse_search_space_rejects_unknown_type() -> None:
    with pytest.raises(SearchSpaceError, match="unsupported type"):
        parse_search_space(
            {
                "search_space": {
                    "lr": {"type": "model", "path": "training.lr", "values": [1e-4]},
                }
            }
        )


def test_parse_search_space_rejects_malformed_dotted_path() -> None:
    with pytest.raises(SearchSpaceError, match="valid dotted path"):
        parse_search_space(
            {
                "search_space": {
                    "temperature": {"path": "models..temperature", "values": [0.0]},
                }
            }
        )


def test_range_bounds_must_be_numeric() -> None:
    with pytest.raises(SearchSpaceError, match="must be numbers"):
        SearchSpaceSpec.from_mapping(
            "temperature",
            {"type": "fabric", "path": "models.default.temperature", "low": "0.1", "high": "0.9"},
        )
    with pytest.raises(SearchSpaceError, match="must be numbers"):
        SearchSpaceSpec.from_mapping(
            "temperature",
            {"type": "fabric", "path": "models.default.temperature", "low": True, "high": False},
        )


def test_grid_trial_count_is_cartesian_product() -> None:
    space = parse_search_space(
        {
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "low": 0.0,
                    "high": 0.4,
                    "step": 0.2,
                },
                "top_p": {
                    "type": "fabric",
                    "path": "models.default.top_p",
                    "values": [0.7, 0.85, 1.0],
                },
            }
        }
    )
    assert grid_trial_count(space) == 3 * 3


def test_suggestions_by_path_maps_logical_names() -> None:
    space = parse_search_space(
        {
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "values": [0.0, 0.2],
                }
            }
        }
    )
    assert suggestions_by_path(space, {"temperature": 0.2}) == {"models.default.temperature": 0.2}
