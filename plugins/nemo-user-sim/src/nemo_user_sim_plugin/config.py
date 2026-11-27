# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin settings, read from the ``user_sim`` block of the platform config file."""

from typing import Any, ClassVar

from nemo_helix_plugin.config import NemoConfig


class UserSimConfig(NemoConfig):
    """Model fields are gateway model entity names, as ``nemo models list`` shows them."""

    plugin_name: ClassVar[str] = "user_sim"
    plugin_description: ClassVar[str] = "UserSim plugin settings"

    user_model: str = "openai-gpt-oss-20b"
    judge_model: str = "nvidia-nemotron-3-5-lightning-30b-a3b"
    summary_model: str = "openai-gpt-oss-20b"
    evaluator_model: str = "nvidia-nemotron-3-5-lightning-30b-a3b"
    embedding_model: str = "nvidia-nemotron-3-embed-1b"
    max_concurrent_conversations: int = 4
    # Extra settings per UserSim role, passed through to rows of UserSim's models file. UserSim caps judge
    # output at 4,096 tokens, and a thinking judge spends most of that before writing its JSON, so the
    # default turns thinking off for the evaluator. Clear it when the evaluator model doesn't take this knob.
    model_settings: dict[str, dict[str, Any]] = {
        "evaluator_model": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    }
    personas_dir: str | None = None
