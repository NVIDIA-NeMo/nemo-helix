# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Checkpoint processing for Automodel backend.

This module handles:
- Finding the best checkpoint after training
- LoRA adapter merging
- Chat template preservation
- FSDP2 architecture fix
- HF export and format conversion
- ONNX export for embedding models

Supports both LLM and embedding (biencoder) models through unified functions.
"""

import json
import logging
import re
import shutil
from enum import StrEnum
from pathlib import Path

from nhx.automodel.tasks.training.chat_templates import (
    apply_chat_template_to_checkpoint,
    resolve_chat_template,
)
from nhx.automodel.tasks.training.schemas import (
    CheckpointFormat,
    CheckpointInfo,
    FinetuningType,
    Precision,
    TrainingStepConfig,
)

logger = logging.getLogger(__name__)


class ModelType(StrEnum):
    """Type of model for checkpoint processing."""

    LLM = "llm"
    EMBEDDING = "embedding"
    CROSS_ENCODER = "cross_encoder"


def extract_precision_from_model_config(model_path: str | Path) -> Precision | None:
    """
    Extract precision from a HuggingFace model's config.json.

    HuggingFace models store their torch_dtype in config.json (e.g., "bfloat16").
    This function reads that value and maps it to our Precision enum.

    This is used to determine the actual training precision when "auto" was used
    for torch_dtype. The precision comes from the base model's config, not from
    the output checkpoint (which may only contain adapter weights for LoRA).

    Args:
        model_path: Path to the model directory containing config.json

    Returns:
        Precision enum value if found, None otherwise
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        logger.warning(f"config.json not found at {config_path}, cannot extract precision")
        return None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        torch_dtype = config.get("torch_dtype")
        if torch_dtype is None:
            logger.warning("torch_dtype not found in config.json")
            return None

        try:
            precision = Precision.from_hf_dtype(torch_dtype)
            logger.info(f"Extracted precision from model config: {torch_dtype} -> {precision.value}")
            return precision
        except ValueError:
            logger.warning(f"Unknown torch_dtype '{torch_dtype}' in config.json, cannot map to Precision")
            return None

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read config.json: {e}")
        return None


def extract_step_number(path: Path) -> int:
    """Extract step number from directory name like 'epoch_0_step_99'"""
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def get_model_dir_from_checkpoint(checkpoint_dir: Path, is_peft: bool) -> Path:
    """
    Extract model directory from checkpoint directory.
    """
    if is_peft:
        # For LoRA, checkpoint is saved directly under model/ directory
        model_dir = checkpoint_dir / "model"
        if model_dir.exists() and model_dir.is_dir():
            logger.info(f"Found LoRA checkpoint at: {model_dir}")
            return model_dir.resolve()
    else:
        # For full-sft, check for consolidated directory first
        consolidated_dir = checkpoint_dir / "model" / "consolidated"
        if consolidated_dir.exists() and consolidated_dir.is_dir():
            logger.info(f"Found consolidated checkpoint at: {consolidated_dir}")
            return consolidated_dir.resolve()

        # Fallback to model/ directory if consolidated doesn't exist
        model_dir = checkpoint_dir / "model"
        if model_dir.exists() and model_dir.is_dir():
            logger.info(f"Found sharded checkpoint at: {model_dir}")
            return model_dir.resolve()

    raise FileNotFoundError(f"Model directory not found in checkpoint {checkpoint_dir}")


def find_best_checkpoint(
    workspace_dir: Path,
    config: TrainingStepConfig,
    model_type: ModelType = ModelType.LLM,
) -> Path:
    """
    Find the best checkpoint directory.
    """
    base_dir = workspace_dir / "checkpoints"
    is_peft = config.training.finetuning_type in (FinetuningType.LORA, FinetuningType.LORA_MERGED)
    type_label = "" if model_type == ModelType.LLM else model_type.value

    # Order of preference:
    # 1. LOWEST_VAL symlink
    # 2. LATEST symlink
    # 3. Highest step number

    for link_name in ["LOWEST_VAL", "LATEST"]:
        link = base_dir / link_name
        if link.exists() and link.is_symlink():
            try:
                target = link.resolve()
                if target.exists():
                    logger.info(f"Using {link_name} {type_label} checkpoint: {target.name}".replace("  ", " "))
                    return get_model_dir_from_checkpoint(target, is_peft)
            except Exception as e:
                logger.warning(f"Failed to resolve {link_name} symlink: {e}")

    # Fallback: scan directories
    epoch_step_dirs = list(base_dir.glob("epoch_*_step_*"))
    if not epoch_step_dirs:
        raise FileNotFoundError(f"No {type_label} checkpoint directories found in {base_dir}".replace("  ", " "))

    best_checkpoint = max(epoch_step_dirs, key=extract_step_number)
    logger.info(f"Using latest {type_label} checkpoint by step number: {best_checkpoint.name}".replace("  ", " "))
    return get_model_dir_from_checkpoint(best_checkpoint, is_peft)


def fix_fsdp2_architecture(model_path: Path) -> None:
    """
    Fix FSDP2 architecture naming issue in HuggingFace config.

    FSDP2 adds "FSDP" prefix to architecture names (e.g., "FSDPLlamaForCausalLM"
    instead of "LlamaForCausalLM"). This function removes that prefix to ensure
    the checkpoint is compatible with standard HuggingFace/vLLM loading.

    Reference: https://github.com/huggingface/transformers/commit/dc262ee6f57f2154f5233e53482da14dbe3be834
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        logger.warning(f"config.json not found at {config_path}, skipping FSDP2 fix")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    if "architectures" not in config:
        return

    original_archs = config["architectures"]
    fixed_archs = [arch.removeprefix("FSDP") for arch in original_archs]

    if original_archs != fixed_archs:
        config["architectures"] = fixed_archs
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Fixed FSDP2 architecture names: {original_archs} -> {fixed_archs}")


def merge_lora_adapter(
    adapter_path: Path,
    base_model_path: str,
    output_path: Path,
) -> None:
    """
    Merge LoRA adapter weights into the base model.

    Uses HuggingFace's PEFT library to:
    1. Load the base model
    2. Attach the LoRA adapter
    3. Merge weights using merge_and_unload()
    4. Save as a standard HuggingFace checkpoint

    Note: This function only supports LLM models. For embedding models,
    use merge_lora_embedding_adapter() instead.

    Args:
        adapter_path: Path to the LoRA adapter checkpoint
        base_model_path: Path to the base model (for loading weights)
        output_path: Where to save the merged model
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "LoRA merge requires 'peft' and 'transformers' packages. Ensure they are installed in the container."
        ) from e

    logger.info(f"Merging LoRA adapter from {adapter_path} with base model {base_model_path}")

    # Use scratch directory if available for better I/O performance
    tmp_path = Path("/scratch/merged_lora") if Path("/scratch").is_dir() else Path("/tmp/merged_lora")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load base model in mergeable dtype (not quantized)
        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 2. Attach the LoRA adapter
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, str(adapter_path))

        # 3. Merge LoRA weights into base model
        logger.info("Merging LoRA weights...")
        model = model.merge_and_unload()

        # 4. Save merged model
        logger.info(f"Saving merged model to {tmp_path}...")
        model.save_pretrained(tmp_path, safe_serialization=True)

        # 5. Save tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(tmp_path)

        # 6. Copy to output path
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path, output_path, dirs_exist_ok=True)

        logger.info(f"Successfully merged LoRA adapter to {output_path}")

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp_path, ignore_errors=True)


def merge_lora_embedding_adapter(
    adapter_path: Path,
    base_model_path: str,
    output_path: Path,
) -> None:
    """Merge a LoRA adapter into a base embedding model.

    This intentionally mirrors the logic in Automodel's `tools/merge_lora.py`,
    but is implemented locally because the customizer container may not have
    that module on `PYTHONPATH`.

    Args:
        adapter_path: Path to the PEFT adapter directory.
        base_model_path: HuggingFace model name or path for the base encoder.
        output_path: Where to write the merged model.
    """
    try:
        import gc

        import torch
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "LoRA merge requires 'peft' and 'transformers' packages. Ensure they are installed in the container."
        ) from e

    logger.info("Merging embedding LoRA adapter from %s with base model %s", adapter_path, base_model_path)

    # Use scratch directory if available for better I/O performance
    tmp_path = Path("/scratch/merged_lora") if Path("/scratch").is_dir() else Path("/tmp/merged_lora")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = None
    try:
        logger.info("Loading base model (AutoModel): %s", base_model_path)
        model = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))

        logger.info("Merging adapter into base model")
        model = model.merge_and_unload()

        logger.info("Saving merged model to %s", tmp_path)
        model.save_pretrained(tmp_path, safe_serialization=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            tokenizer.save_pretrained(tmp_path)
            logger.info("Tokenizer saved to %s", tmp_path)
        except Exception as e:
            logger.warning("Could not save tokenizer: %s", e)

        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path, output_path, dirs_exist_ok=True)
        logger.info("Successfully merged embedding LoRA adapter to %s", output_path)

    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
        torch.cuda.empty_cache()
        gc.collect()


def merge_lora_cross_encoder_adapter(
    adapter_path: Path,
    base_model_path: str,
    output_path: Path,
) -> None:
    """Merge a LoRA adapter into a cross-encoder (sequence-classification) base model."""
    try:
        import gc

        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "LoRA merge requires 'peft' and 'transformers' packages. Ensure they are installed in the container."
        ) from e

    logger.info("Merging cross-encoder LoRA adapter from %s with base model %s", adapter_path, base_model_path)

    tmp_path = Path("/scratch/merged_lora") if Path("/scratch").is_dir() else Path("/tmp/merged_lora")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = None
    try:
        logger.info("Loading base model (AutoModelForSequenceClassification): %s", base_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))

        logger.info("Merging adapter into base model")
        model = model.merge_and_unload()

        logger.info("Saving merged model to %s", tmp_path)
        model.save_pretrained(tmp_path, safe_serialization=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            tokenizer.save_pretrained(tmp_path)
            logger.info("Tokenizer saved to %s", tmp_path)
        except Exception as e:
            logger.warning("Could not save tokenizer: %s", e)

        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path, output_path, dirs_exist_ok=True)
        logger.info("Successfully merged cross-encoder LoRA adapter to %s", output_path)

    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
        torch.cuda.empty_cache()
        gc.collect()


def export_onnx(
    model_path: Path,
    output_path: Path,
    tokenizer_path: str,
) -> Path:
    """Export an embedding model to ONNX format.

    Uses Automodel's export_to_onnx to export to ONNX format.
    The resulting `model.onnx` is written into *output_path* alongside
    the existing HuggingFace checkpoint files.

    Args:
        model_path: Path to the HuggingFace model directory (config.json + weights).
        output_path: Directory where ``model.onnx`` will be written.
        tokenizer_path: Fallback tokenizer location (base model path). Used when
            the checkpoint directory does not contain tokenizer files.

    Returns:
        Path to the exported ``model.onnx`` file.
    """
    # need to import here for the tests
    # llama_bidirectional is Automodel's module path; export_to_onnx uses HF AutoModel and works for any encoder checkpoint.
    from nemo_automodel.components.models.llama_bidirectional.export_onnx import export_to_onnx

    logger.info(f"Exporting embedding model at path {model_path} to ONNX format at path {output_path}")

    try:
        onnx_path = export_to_onnx(
            model_path=str(model_path),
            output_dir=str(output_path),
            tokenizer_path=tokenizer_path,
            pooling="avg",
            normalize=True,
            opset=18,
            export_dtype="fp16",
            verify=True,
        )
    except Exception:
        logger.exception(f"ONNX export failed for model at {model_path}")
        raise

    logger.info(f"ONNX model exported to {onnx_path}")
    return Path(onnx_path)


_ONNX_TOP_LEVEL_PATTERNS = {"model.onnx", "model.onnx.data", "tokenizer"}


def _restructure_embedding_output(output_path: Path) -> None:
    """Move HF artifacts into ``alternates/hf/`` so the NIM selects the ONNX profile.

    NIM scans the top-level directory to choose the model backend.  If it sees
    ``.safetensors`` files it creates a PyTorch profile, which is unsupported
    for custom models in many NIM versions.  The legacy customizer kept only
    ``model.onnx`` (+ tokenizer/) at the root and placed HF weights under
    ``alternates/hf/``.  This function reproduces that layout.
    """
    alternates_hf = output_path / "alternates" / "hf"
    alternates_hf.mkdir(parents=True, exist_ok=True)

    for entry in list(output_path.iterdir()):
        if entry.name in _ONNX_TOP_LEVEL_PATTERNS or entry.name == "alternates":
            continue
        dest = alternates_hf / entry.name
        logger.info("Moving %s -> %s", entry, dest)
        shutil.move(str(entry), str(dest))

    logger.info("Restructured embedding output: ONNX at top level, HF in alternates/hf/")


def process_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    customizer_config: TrainingStepConfig,
    model_type: ModelType = ModelType.LLM,
    resolved_chat_template: str | None = None,
) -> CheckpointInfo:
    """
    Process checkpoint to standard output format.

    Works for both LLM and embedding (biencoder) models.

    Handles three scenarios:
    1. Full weights training: Copy checkpoint, fix FSDP2 arch, preserve chat template (LLM only)
    2. LoRA (unmerged): Copy adapter, preserve format as hf-peft
    3. LoRA merged: Merge adapter with base model, output as standard HF

    Args:
        checkpoint_path: Path to the checkpoint directory (model files)
        output_path: Where to write the processed checkpoint
        customizer_config: Training configuration with model paths and settings
        model_type: Type of model ("llm" or "embedding")
        resolved_chat_template: Pre-resolved chat template from training config (LLM only).
            If provided, this template is used. Otherwise, falls back to
            priority-based resolution using model.name and model.path.

    Returns:
        CheckpointInfo with output path, format, and precision
    """
    output_path.mkdir(parents=True, exist_ok=True)

    finetuning_type = customizer_config.training.finetuning_type
    base_model_path = customizer_config.model.path
    is_embedding = model_type == ModelType.EMBEDDING
    is_cross_encoder = model_type == ModelType.CROSS_ENCODER
    is_llm = model_type == ModelType.LLM
    type_label = "" if is_llm else model_type.value

    # Resolve chat template using the same priority logic as training:
    # 1. Use pre-resolved template if provided (ensures consistency with training)
    # 2. Otherwise, resolve using priority-based selection
    chat_template: str | None = None
    if is_llm:
        if resolved_chat_template is not None:
            chat_template = resolved_chat_template
            logger.info("Using pre-resolved chat template from training config")
        else:
            # Fall back to priority-based resolution (user_template from fileset metadata takes priority)
            chat_template = resolve_chat_template(
                model_path=base_model_path,
                model_name=customizer_config.model.name,
                user_template=customizer_config.model.chat_template,
            )

    if finetuning_type == FinetuningType.LORA_MERGED:
        # LoRA merged: merge adapter weights into base model
        # For embedding models, this produces a full-weight model compatible with ONNX export and NIM serving.
        if is_embedding:
            merge_lora_embedding_adapter(
                adapter_path=checkpoint_path,
                base_model_path=base_model_path,
                output_path=output_path,
            )
        elif is_cross_encoder:
            merge_lora_cross_encoder_adapter(
                adapter_path=checkpoint_path,
                base_model_path=base_model_path,
                output_path=output_path,
            )
        else:
            merge_lora_adapter(
                adapter_path=checkpoint_path,
                base_model_path=base_model_path,
                output_path=output_path,
            )
        checkpoint_format = CheckpointFormat.HF

        # Fix FSDP2 architecture naming
        fix_fsdp2_architecture(output_path)
        # Apply chat template for LLM models only
        if chat_template:
            apply_chat_template_to_checkpoint(output_path, chat_template)

    elif finetuning_type == FinetuningType.LORA:
        # LoRA unmerged: just copy the adapter files
        logger.info(f"Copying {type_label} LoRA adapter from {checkpoint_path} to {output_path}".replace("  ", " "))
        shutil.copytree(checkpoint_path, output_path, dirs_exist_ok=True)
        checkpoint_format = CheckpointFormat.HF_PEFT
        # Note: For hf-peft, chat template is inherited from base model at inference time

    else:
        # Full weights training: copy and process
        logger.info(
            f"Copying {type_label} full weights checkpoint from {checkpoint_path} to {output_path}".replace("  ", " ")
        )
        shutil.copytree(checkpoint_path, output_path, dirs_exist_ok=True)
        checkpoint_format = CheckpointFormat.HF

        # Fix FSDP2 architecture naming
        fix_fsdp2_architecture(output_path)
        # Apply chat template for LLM models only
        if chat_template:
            apply_chat_template_to_checkpoint(output_path, chat_template)

    if is_embedding:
        export_onnx(
            model_path=output_path,
            output_path=output_path,
            tokenizer_path=base_model_path,
        )
        _restructure_embedding_output(output_path)

    # Determine precision: use explicit config value, or extract from base model
    precision = customizer_config.model.precision
    if precision is None:
        precision = extract_precision_from_model_config(customizer_config.model.path)

    return CheckpointInfo(
        path=str(output_path),
        format=checkpoint_format,
        precision=precision,
    )
