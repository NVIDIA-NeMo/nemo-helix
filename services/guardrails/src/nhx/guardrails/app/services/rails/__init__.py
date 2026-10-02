# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter
from nemoguardrails.llm.providers import register_llm_provider, register_provider
from nemoguardrails.types import LLMModel
from nhx.guardrails.app.constants import NIM_CHAT, NIM_LLM
from nhx.guardrails.app.llms.chat.nim import ChatNIM
from nhx.guardrails.app.llms.completion.nim import NIM

log = logging.getLogger(__name__)


def _init_nim_chat_model(*, model: str, **kwargs: Any) -> LLMModel:
    """Create a NeMo Guardrails model backed by the platform's ChatNIM client."""
    return LangChainLLMAdapter(ChatNIM(model=model, **kwargs))


def register_providers():
    """Register Chat/LLM providers to NeMo Guardrails."""

    register_provider(NIM_CHAT, _init_nim_chat_model)
    register_llm_provider(NIM_LLM, NIM)


register_providers()
