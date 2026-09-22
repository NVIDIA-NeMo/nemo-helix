# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from langchain_core.language_models import BaseChatModel


def apply_langchain_patch():
    """Patching various LangChain bits until they get fixed."""

    # This method returns {} all the time in the original implementation
    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return llm_outputs[0] if len(llm_outputs) > 0 else {}

    BaseChatModel._combine_llm_outputs = _combine_llm_outputs
