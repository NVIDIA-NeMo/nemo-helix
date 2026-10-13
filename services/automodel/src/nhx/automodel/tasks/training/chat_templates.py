# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chat template resolution and application for training backends.

This module provides:
1. Priority-based chat template selection (resolve_chat_template)
2. Applying chat templates to output checkpoints (apply_chat_template_to_checkpoint)

Chat template priority order:
1. User-provided template (via API)
2. Custom template from DEFAULT_CHAT_TEMPLATES map (enhanced for tool calling)
3. Model's built-in tokenizer template (fallback)

The custom templates in the templates/ directory extend base model templates with:
- Tool calling support: <TOOLCALL>, <AVAILABLE_TOOLS>, <TOOL_RESPONSE> formatting
- Generation markers: {% generation %}...{% endgeneration %} blocks for loss masking
- Enhanced compatibility across models
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory containing custom chat template jinja files
TEMPLATES_DIR = Path(__file__).parent / "templates"

# ============================================================================
# Model Name Constants
# ============================================================================

# Meta Llama models
META_LLAMA_31_8B_INSTRUCT = "meta/llama-3.1-8b-instruct"
META_LLAMA_31_70B_INSTRUCT = "meta/llama-3.1-70b-instruct"
META_LLAMA_31_405B_INSTRUCT = "meta/llama-3.1-405b-instruct"
META_LLAMA_32_1B = "meta/llama-3.2-1b"
META_LLAMA_32_1B_INSTRUCT = "meta/llama-3.2-1b-instruct"
META_LLAMA_32_3B_INSTRUCT = "meta/llama-3.2-3b-instruct"
META_LLAMA_33_70B_INSTRUCT = "meta/llama-3.3-70b-instruct"
# NVIDIA Nemotron models
NVIDIA_NEMOTRON_31_8B = "nvidia/nemotron-nano-llama-3.1-8b"
NVIDIA_NEMOTRON_31_70B = "nvidia/nemotron-llama-3.1-70b"
NVIDIA_NEMOTRON_33_49B = "nvidia/nemotron-super-llama-3.3-49b"
NVIDIA_NEMOTRON_33_49B_V1_5 = "nvidia/nemotron-super-llama-3.3-49b-v1.5"
# NIM model names (alternative naming)
NIM_NVIDIA_NEMOTRON_31_8B = "nvidia/llama-3.1-nemotron-nano-8b-v1"
NIM_NVIDIA_NEMOTRON_31_70B = "nvidia/llama-3.1-nemotron-70b-instruct"
NIM_NVIDIA_NEMOTRON_33_49B = "nvidia/llama-3.3-nemotron-super-49b-v1"
NIM_NVIDIA_NEMOTRON_33_49B_V1_5 = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
# Microsoft models
PHI_4 = "microsoft/phi-4"

# ============================================================================
# Default Chat Templates Map
# ============================================================================

# Maps model names to custom jinja template filenames.
# These templates extend the base model templates with:
# - Tool calling support
# - Generation markers for loss masking
# - Enhanced compatibility
DEFAULT_CHAT_TEMPLATES: dict[str, str] = {
    # Llama 3.1 family
    META_LLAMA_31_8B_INSTRUCT: "llama-3.1-instruct.jinja",
    META_LLAMA_31_70B_INSTRUCT: "llama-3.1-instruct.jinja",
    META_LLAMA_31_405B_INSTRUCT: "llama-3.1-instruct.jinja",
    # Llama 3.2 family
    META_LLAMA_32_1B: "llama-3.2-instruct.jinja",
    META_LLAMA_32_1B_INSTRUCT: "llama-3.2-instruct.jinja",
    META_LLAMA_32_3B_INSTRUCT: "llama-3.2-instruct.jinja",
    # Llama 3.3 family
    META_LLAMA_33_70B_INSTRUCT: "llama-3.3-instruct.jinja",
    # Nemotron family
    NVIDIA_NEMOTRON_31_8B: "nemotron-3.1.jinja",
    NVIDIA_NEMOTRON_31_70B: "nemotron-3.1.jinja",
    NVIDIA_NEMOTRON_33_49B: "nemotron-super-3.3.jinja",
    NVIDIA_NEMOTRON_33_49B_V1_5: "nemotron-super-3.3.jinja",
    # NIM Nemotron (alternative naming)
    NIM_NVIDIA_NEMOTRON_31_8B: "nemotron-3.1.jinja",
    NIM_NVIDIA_NEMOTRON_31_70B: "nemotron-3.1.jinja",
    NIM_NVIDIA_NEMOTRON_33_49B: "nemotron-super-3.3.jinja",
    NIM_NVIDIA_NEMOTRON_33_49B_V1_5: "nemotron-super-3.3.jinja",
    # Microsoft
    PHI_4: "phi-4.jinja",
}


def _load_template_file(template_filename: str) -> str | None:
    """Load a custom template from the templates directory."""
    template_path = TEMPLATES_DIR / template_filename
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    logger.warning(f"Template file not found: {template_path}")
    return None


def _get_tokenizer_chat_template(model_path: str) -> str | None:
    """
    Get chat template from model's tokenizer.

    Uses AutoTokenizer which handles all model formats (HF, NeMo, custom).
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        template = getattr(tokenizer, "chat_template", None)
        if template:
            logger.debug(f"Found chat template in tokenizer for {model_path}")
        return template
    except Exception as e:
        logger.warning(f"Could not load tokenizer to get chat template: {e}")
        return None


def resolve_chat_template(
    model_path: str,
    model_name: str | None = None,
    user_template: str | None = None,
) -> str | None:
    """
    Resolve chat template using priority-based selection.

    Priority order:
    1. User-provided template (highest priority)
    2. Custom template from DEFAULT_CHAT_TEMPLATES (if model_name matches)
    3. Model's built-in tokenizer template (fallback)

    Args:
        model_path: Path to the model directory (for tokenizer fallback).
        model_name: Canonical model name (e.g., "meta/llama-3.1-8b-instruct").
                   Used to look up custom templates.
        user_template: User-provided template string (takes highest priority).

    Returns:
        The resolved chat template string, or None if no template found.
    """
    # Priority 1: User-provided template
    if user_template:
        logger.info("Using user-provided chat template")
        return user_template

    # Priority 2: Custom template from DEFAULT_CHAT_TEMPLATES
    if model_name and model_name in DEFAULT_CHAT_TEMPLATES:
        template_filename = DEFAULT_CHAT_TEMPLATES[model_name]
        template = _load_template_file(template_filename)
        if template:
            logger.info(f"Using custom chat template for {model_name}: {template_filename}")
            return template

    # Priority 3: Model's built-in tokenizer template
    template = _get_tokenizer_chat_template(model_path)
    if template:
        logger.info(f"Using model's built-in chat template from {model_path}")
        return template

    logger.warning(f"No chat template found for model_name={model_name}, model_path={model_path}")
    return None


def apply_chat_template_to_checkpoint(
    output_path: Path,
    chat_template: str | None,
) -> None:
    """
    Apply chat template to the output checkpoint's tokenizer_config.json.

    Also ensures pad_token is set if missing (uses eos_token as fallback),
    which is required by many inference frameworks.

    Args:
        output_path: Path to the checkpoint directory containing tokenizer_config.json.
        chat_template: The chat template string to apply. If None, skips application.
    """
    if not chat_template:
        logger.warning("No chat template provided, skipping")
        return

    tokenizer_config = output_path / "tokenizer_config.json"
    if not tokenizer_config.exists():
        logger.warning(f"tokenizer_config.json not found at {output_path}")
        return

    with open(tokenizer_config, "r") as f:
        config = json.load(f)

    config["chat_template"] = chat_template

    with open(tokenizer_config, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Applied chat template to output checkpoint")
