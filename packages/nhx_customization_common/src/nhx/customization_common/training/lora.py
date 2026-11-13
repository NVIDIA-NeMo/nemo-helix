# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge an HF PEFT adapter into a base HuggingFace model.

Automodel and NeMo-RL both call this at ``lora_merged`` export. ``peft`` and
``transformers`` are imported inside the function so the compiler and the
plugins can import this package without them.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_builtin_model_class(base_model_path: str) -> bool:
    """Whether transformers ships its own class for the checkpoint's model type."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    try:
        model_type = json.loads((Path(base_model_path) / "config.json").read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError, AttributeError):
        return False
    return isinstance(model_type, str) and model_type in CONFIG_MAPPING


def _drop_auto_map(config_path: Path) -> None:
    """A checkpoint saved from transformers' own class ships no remote code to point at."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.pop("auto_map", None) is not None:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _merge_parent() -> Path:
    """Directory that holds one staging folder per merge.

    Prefer the task's ephemeral storage so concurrent jobs on one host do not
    share a directory. Otherwise use node scratch, then /tmp.
    """
    ephemeral = os.environ.get("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH")
    if ephemeral:
        return Path(ephemeral)
    if Path("/scratch").is_dir():
        return Path("/scratch")
    return Path("/tmp")


def merge_lora_adapter(
    adapter_path: Path,
    base_model_path: str,
    output_path: Path,
    trust_remote_code: bool,
    has_vision_tower: bool = False,
    builtin_model_class: bool = False,
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
        builtin_model_class: The adapter uses the module names of transformers' own class for
            this model type. When transformers has that class, load it instead of the
            checkpoint's remote code, whose names can differ (Nemotron-H: ``backbone.*``
            rather than ``model.*``) so that PEFT would not find the adapter's targets.
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

    model_trust_remote_code = trust_remote_code
    if builtin_model_class and trust_remote_code and _has_builtin_model_class(base_model_path):
        model_trust_remote_code = False
        logger.info("Loading the base with transformers' own class, whose module names the adapter uses")

    logger.info("Merging LoRA adapter from %s into base model %s", adapter_path, base_model_path)
    parent = _merge_parent()
    parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(prefix="merged_lora-", dir=parent))
    try:
        logger.info("Loading base model...")
        model = model_cls.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=model_trust_remote_code,
        )
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        logger.info("Merging LoRA weights...")
        model = model.merge_and_unload()
        logger.info("Saving merged model to %s...", tmp_path)
        model.save_pretrained(tmp_path, safe_serialization=True)
        if model_trust_remote_code != trust_remote_code:
            _drop_auto_map(tmp_path / "config.json")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
        if tokenizer is None:
            raise RuntimeError(f"AutoTokenizer.from_pretrained returned no tokenizer for {base_model_path}")
        tokenizer.save_pretrained(tmp_path)
        if has_vision_tower:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
            if processor is None:
                raise RuntimeError(f"AutoProcessor.from_pretrained returned no processor for {base_model_path}")
            processor.save_pretrained(tmp_path)
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path, output_path, dirs_exist_ok=True)
        logger.info("Successfully merged LoRA adapter to %s", output_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
