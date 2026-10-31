# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge an HF PEFT adapter into a base HuggingFace model.

Automodel and NeMo-RL both call this at ``lora_merged`` export. ``peft`` and
``transformers`` are imported inside the function so the compiler and the
plugins can import this package without them.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _merge_workdir() -> Path:
    return Path("/scratch/merged_lora") if Path("/scratch").is_dir() else Path("/tmp/merged_lora")


def merge_lora_adapter(
    adapter_path: Path,
    base_model_path: str,
    output_path: Path,
    trust_remote_code: bool,
    has_vision_tower: bool = False,
) -> None:
    """Merge LoRA adapter weights into the base model.

    Uses HuggingFace's PEFT library to:
    1. Load the base model
    2. Attach the LoRA adapter
    3. Merge weights using merge_and_unload()
    4. Save as a standard HuggingFace checkpoint

    A causal language model loads through ``AutoModelForCausalLM``. A vision
    checkpoint loads through ``AutoModelForImageTextToText``. For embedding
    models, use ``merge_lora_embedding_adapter()`` instead.

    Args:
        adapter_path: Path to the LoRA adapter checkpoint
        base_model_path: Path to the base model (for loading weights)
        output_path: Where to save the merged model
        trust_remote_code: Job flag for loading the *base* fileset.
        has_vision_tower: Load the image-text model class when the base has a vision tower.
    """
    try:
        import torch  # ty: ignore[unresolved-import]
        from peft import PeftModel  # ty: ignore[unresolved-import]
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "LoRA merge requires 'peft' and 'transformers'. They are installed in the training image."
        ) from e

    if has_vision_tower:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    else:
        model_cls = AutoModelForCausalLM

    logger.info("Merging LoRA adapter from %s into base model %s", adapter_path, base_model_path)
    tmp_path = _merge_workdir()
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Loading base model...")
        model = model_cls.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        logger.info("Merging LoRA weights...")
        model = model.merge_and_unload()
        logger.info("Saving merged model to %s...", tmp_path)
        model.save_pretrained(tmp_path, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
        if tokenizer is None:
            raise RuntimeError(f"AutoTokenizer.from_pretrained returned no tokenizer for {base_model_path}")
        tokenizer.save_pretrained(tmp_path)
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path, output_path, dirs_exist_ok=True)
        logger.info("Successfully merged LoRA adapter to %s", output_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
