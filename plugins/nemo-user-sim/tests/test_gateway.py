# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_user_sim_plugin.gateway import write_models_file
from usersim.cli._models import load_models_config


def test_gateway_names_keep_usersim_inference_settings(tmp_path):
    # UserSim keys its per-model settings by provider ID; the gateway serves the same model by entity name.
    path = write_models_file(
        tmp_path / "models.toml",
        gateway_url="http://gateway/v1",
        models={"user_model": "openai-gpt-oss-20b", "judge_model": "nvidia-nemotron-3-5-lightning-30b-a3b"},
    )

    user, judge = load_models_config(path).models

    assert (user.model, user.max_tokens, user.extra_body) == ("openai-gpt-oss-20b", 4096, {"reasoning_effort": "high"})
    assert judge.extra_body is None


def test_provider_prefixed_names_still_get_usersim_settings_and_timeout(tmp_path):
    # inference-api.nvidia.com serves Lightning as nvidia/nvidia/nemotron-3.5-lightning.
    path = write_models_file(
        tmp_path / "models.toml",
        gateway_url="http://gateway/v1",
        models={"evaluator_model": "nvidia-nvidia-nemotron-3-5-lightning"},
    )

    (evaluator,) = load_models_config(path).models

    assert (evaluator.max_tokens, evaluator.timeout) == (4096, 300)


def test_role_settings_from_the_plugin_config_win_over_usersims(tmp_path):
    path = write_models_file(
        tmp_path / "models.toml",
        gateway_url="http://gateway/v1",
        models={"evaluator_model": "nvidia-nemotron-3-5-lightning-30b-a3b"},
        role_settings={"evaluator_model": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}},
    )

    (evaluator,) = load_models_config(path).models

    assert evaluator.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert evaluator.max_tokens == 4096
