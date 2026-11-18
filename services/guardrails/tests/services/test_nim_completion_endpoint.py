# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the text-completions main model uses the endpoint from its own config."""

import contextvars

from nhx.guardrails.app.constants import NIM_LLM
from nhx.guardrails.app.llms.completion.nim import NIM
from nhx.guardrails.app.utils.context_utils import set_main_model_into_context
from nhx.guardrails.entities.values._private import Model


def _build(base_url: str) -> NIM:
    """Build the model the way engine construction does, with its config's main model in context.

    Each build gets its own context, like a separate request would.
    """

    def build() -> NIM:
        set_main_model_into_context(Model(type="main", engine=NIM_LLM, model="m", parameters={"base_url": base_url}))
        return NIM(model="m")

    return contextvars.copy_context().run(build)


def test_each_engine_keeps_its_own_endpoint():
    """Building an engine for a second config shouldn't change where the first one sends requests.

    The endpoint used to be stored on the class, so the last engine built won for everyone.
    """
    config_a = _build("https://config-a.example/v1")
    config_b = _build("https://config-b.example/v1")

    assert config_a._get_request_url() == "https://config-a.example/v1/completions"
    assert config_b._get_request_url() == "https://config-b.example/v1/completions"
