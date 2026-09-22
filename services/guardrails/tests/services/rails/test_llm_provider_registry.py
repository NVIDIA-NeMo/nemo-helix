# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter
from nemoguardrails.llm.models.initializer import init_llm_model
from nemoguardrails.llm.providers import get_chat_provider_names, get_llm_provider_names
from nhx.guardrails.app.constants import NIM_CHAT, NIM_LLM
from nhx.guardrails.app.services import rails

rails.register_providers()


def test_chat_model_registered_in_nemoguardrails():
    assert NIM_CHAT in get_chat_provider_names(), "NIM_CHAT provider not registered in NeMo Guardrails"


def test_llm_model_registered_in_nemoguardrails():
    assert NIM_LLM in get_llm_provider_names(), "NIM_LLM provider not registered in NeMo Guardrails"


def test_chat_model_uses_nemoguardrails_model_interface(monkeypatch):
    chat_nim = object()
    monkeypatch.setattr(rails, "ChatNIM", lambda **_kwargs: chat_nim)

    model = init_llm_model(
        model_name="default/test-model",
        provider_name=NIM_CHAT,
        kwargs={},
    )

    assert isinstance(model, LangChainLLMAdapter)
    assert model._llm is chat_nim
