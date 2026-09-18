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
- ONNX export for embedding and cross-encoder models

Supports LLM, embedding (biencoder), and cross-encoder models through unified functions.
"""

import json
import logging
import re
import shutil
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from nmp.automodel.tasks.training.chat_templates import (
    apply_chat_template_to_checkpoint,
    resolve_chat_template,
)
from nmp.automodel.tasks.training.schemas import (
    CheckpointFormat,
    CheckpointInfo,
    CheckpointSelection,
    ExportConfig,
    FinetuningType,
    Precision,
    TrainingStepConfig,
)

logger = logging.getLogger(__name__)

CHECKPOINT_STATS_FILENAME = "checkpoint_stats.json"


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


def _resolve_checkpoint_link(
    workspace_dir: Path,
    config: TrainingStepConfig,
    link_names: tuple[str, ...],
    model_type: ModelType = ModelType.LLM,
) -> Path:
    """Resolve Automodel checkpoint symlinks in order, then fall back to the highest step."""
    base_dir = workspace_dir / "checkpoints"
    is_peft = config.training.finetuning_type in (FinetuningType.LORA, FinetuningType.LORA_MERGED)
    type_label = "" if model_type == ModelType.LLM else model_type.value

    for link_name in link_names:
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

    latest_checkpoint = max(epoch_step_dirs, key=extract_step_number)
    logger.info(f"Using latest {type_label} checkpoint by step number: {latest_checkpoint.name}".replace("  ", " "))
    return get_model_dir_from_checkpoint(latest_checkpoint, is_peft)


def find_selected_checkpoints(
    workspace_dir: Path,
    config: TrainingStepConfig,
    model_type: ModelType = ModelType.LLM,
) -> dict[str, Path]:
    """Resolve the configured checkpoint selection in publication order."""
    selection = config.schedule.checkpoint_selection
    if selection == CheckpointSelection.BEST:
        return {"best": _resolve_checkpoint_link(workspace_dir, config, ("LOWEST_VAL", "LATEST"), model_type)}
    if selection == CheckpointSelection.LAST:
        return {"last": _resolve_checkpoint_link(workspace_dir, config, ("LATEST",), model_type)}
    return {
        "best": _resolve_checkpoint_link(workspace_dir, config, ("LOWEST_VAL", "LATEST"), model_type),
        "last": _resolve_checkpoint_link(workspace_dir, config, ("LATEST",), model_type),
    }


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


_ENCODER_KEEP_KEYS = ("is_causal", "pooling", "temperature")


def sanitize_encoder_hf(output_path: Path, base_model_path: Path) -> None:
    """Rewrite a published encoder HF tree to the base architecture.

    Restores ``model_type`` and ``architectures`` from the base checkpoint, drops
    ``auto_map``, and deletes Python files in *output_path* that the base tree
    does not ship. Encoder serving fields (``is_causal``, ``pooling``,
    ``temperature``) already on the trained config are preserved.

    Callers must skip this when the job has ``trust_remote_code=true``. This
    helper does not read that flag. Custom files remain required to *trace* ONNX
    from a training checkpoint; sanitize only the published HF directory.

    Args:
        output_path: HuggingFace checkpoint directory to rewrite (fileset root
            before restructure, or ``alternates/hf`` for an ONNX-primary fileset).
        base_model_path: Base HuggingFace directory whose architecture identity
            is restored.
    """
    output_path = Path(output_path)
    base_model_path = Path(base_model_path)
    output_config_path = output_path / "config.json"
    base_config_path = base_model_path / "config.json"
    if not output_config_path.exists() or not base_config_path.exists():
        logger.warning(
            "Skipping encoder HF sanitize: missing config.json (output=%s exists=%s, base=%s exists=%s)",
            output_config_path,
            output_config_path.exists(),
            base_config_path,
            base_config_path.exists(),
        )
        return

    with open(output_config_path) as f:
        config = json.load(f)
    with open(base_config_path) as f:
        base_config = json.load(f)

    kept = {key: config[key] for key in _ENCODER_KEEP_KEYS if key in config}
    if "model_type" in base_config:
        config["model_type"] = base_config["model_type"]
    if "architectures" in base_config:
        config["architectures"] = base_config["architectures"]
    config.pop("auto_map", None)
    config.update(kept)

    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    base_py = {path.relative_to(base_model_path) for path in base_model_path.rglob("*.py") if path.is_file()}
    for py_file in output_path.rglob("*.py"):
        if not py_file.is_file():
            continue
        relative = py_file.relative_to(output_path)
        if relative not in base_py:
            py_file.unlink()
            logger.info("Removed encoder custom code not present in the base checkpoint: %s", relative)

    logger.info(
        "Sanitized encoder HF checkpoint at %s to model_type=%s architectures=%s",
        output_path,
        config.get("model_type"),
        config.get("architectures"),
    )


def merge_lora_adapter(
    adapter_path: Path,
    base_model_path: str,
    output_path: Path,
    trust_remote_code: bool,
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
        trust_remote_code: Job flag for loading the *base* fileset.
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
            trust_remote_code=trust_remote_code,
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
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
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
    trust_remote_code: bool,
) -> None:
    """Merge a LoRA adapter into a base embedding model.

    This intentionally mirrors the logic in Automodel's `tools/merge_lora.py`,
    but is implemented locally because the customizer container may not have
    that module on `PYTHONPATH`.

    Args:
        adapter_path: Path to the PEFT adapter directory.
        base_model_path: HuggingFace model name or path for the base encoder.
        output_path: Where to write the merged model.
        trust_remote_code: Job flag for loading the *base* fileset.
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
            trust_remote_code=trust_remote_code,
        )

        logger.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))

        logger.info("Merging adapter into base model")
        model = model.merge_and_unload()

        logger.info("Saving merged model to %s", tmp_path)
        model.save_pretrained(tmp_path, safe_serialization=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
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
    trust_remote_code: bool,
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
            trust_remote_code=trust_remote_code,
        )

        logger.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))

        logger.info("Merging adapter into base model")
        model = model.merge_and_unload()

        logger.info("Saving merged model to %s", tmp_path)
        model.save_pretrained(tmp_path, safe_serialization=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
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


_EXPORT_SAMPLES = {
    ModelType.EMBEDDING: ["query:hello world", "passage:an example sentence for tracing"],
    ModelType.CROSS_ENCODER: [
        "question:hello \n \n passage:world",
        "question:example \n \n passage:sentence for tracing",
    ],
}

_TORCH_DTYPES = {
    "fp32": "float32",
    "fp16": "float16",
}


def _probe_bidirectional_mask(mod):
    """Read ``create_bidirectional_mask`` off *mod*, or ``None`` if it has none.

    ``getattr(mod, name, None)`` is not enough here. This sweeps every entry in
    ``sys.modules``, and a package using PEP 562 lazy submodule loading decides
    for itself what a miss raises -- ``nemo_automodel.components.models`` raises
    ``ModuleNotFoundError``, which is an ``ImportError`` and so passes straight
    through the ``getattr`` default. Any exception from a foreign ``__getattr__``
    means the same thing for our purposes: this module does not have the mask.
    """
    try:
        return getattr(mod, "create_bidirectional_mask", None)
    except Exception:
        return None


@contextmanager
def _onnx_safe_bidirectional_mask():
    """Replace transformers' SDPA bidirectional mask with an ONNX-traceable additive mask."""
    import torch

    def _mask(config=None, inputs_embeds=None, attention_mask=None, input_embeds=None, **kwargs):
        embeds = inputs_embeds if inputs_embeds is not None else input_embeds
        dtype = embeds.dtype
        batch_size, seq_length, _ = embeds.shape
        mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
        return (1.0 - mask.to(dtype)) * torch.finfo(dtype).min

    patched: dict[str, object] = {}
    for mod_name, mod in list(sys.modules.items()):
        fn = _probe_bidirectional_mask(mod)
        if callable(fn):
            patched[mod_name] = fn
            setattr(mod, "create_bidirectional_mask", _mask)
    try:
        yield
    finally:
        for mod_name, orig in patched.items():
            if mod_name in sys.modules:
                setattr(sys.modules[mod_name], "create_bidirectional_mask", orig)


def _build_export_module(inner, model_type: ModelType, cfg: ExportConfig):
    """Pool embeddings or emit logits, matching the NIM graph contract."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class _CrossEncoderForExport(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            return self.model(**kwargs).logits

    class _EmbeddingForExport(nn.Module):
        def __init__(self, model, pooling: str, normalize: bool):
            super().__init__()
            self.model = model
            self.pooling = pooling
            self.normalize = normalize

        def _pool(self, hidden, attention_mask):
            valid_tokens = attention_mask.bool()
            masked = hidden.masked_fill(~valid_tokens[..., None], 0.0)
            if self.pooling == "avg":
                return masked.sum(dim=1) / (attention_mask.sum(dim=1)[..., None] + 1e-9)

            # Define cls as the first valid token and last as the final valid
            # token. Position reductions work for either tokenizer padding side
            # and avoid data-dependent nonzero indices in the ONNX graph.
            positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(attention_mask)
            if self.pooling == "cls":
                token_index = valid_tokens.to(torch.int64).argmax(dim=1)
            else:
                token_index = positions.masked_fill(~valid_tokens, -1).argmax(dim=1)
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), token_index]

        def forward(self, input_ids, attention_mask, dimensions=None):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs["last_hidden_state"]
            embeddings = self._pool(hidden, attention_mask)
            if dimensions is not None:
                # Truncate to `dimensions`, then L2-renormalize.
                keep = torch.arange(embeddings.shape[1], device=embeddings.device)[None, :] < dimensions[:, None]
                embeddings = embeddings * keep.to(embeddings.dtype)
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings

    if model_type == ModelType.CROSS_ENCODER:
        return _CrossEncoderForExport(inner)
    return _EmbeddingForExport(inner, pooling=cfg.pooling, normalize=cfg.normalize)


def export_onnx(
    model_path: Path,
    output_path: Path,
    tokenizer_path: str,
    model_type: ModelType = ModelType.EMBEDDING,
    cfg: ExportConfig | None = None,
    trust_remote_code: bool = False,
    load_trust_remote_code: bool = False,
) -> Path:
    """Write ``model.onnx`` and ``tokenizer/``. Optionally verify against the traced module."""
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    cfg = cfg or ExportConfig()
    is_cross_encoder = model_type == ModelType.CROSS_ENCODER
    logger.info("Exporting %s at %s to ONNX at %s", model_type.value, model_path, output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    torch_dtype = getattr(torch, _TORCH_DTYPES[cfg.precision])

    loader = AutoModelForSequenceClassification if is_cross_encoder else AutoModel
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    inner = loader.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=load_trust_remote_code,
    ).eval()

    export_model = _build_export_module(inner, model_type, cfg)
    device = torch.device("cuda")
    export_model = export_model.to(device=device, dtype=torch_dtype).eval()

    tokenized = tokenizer(_EXPORT_SAMPLES[model_type], return_tensors="pt", padding=True, truncation=True)
    args = [tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)]
    input_names = ["input_ids", "attention_mask"]
    output_name = "logits" if is_cross_encoder else "embeddings"
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        output_name: {0: "batch_size", 1: "num_labels" if is_cross_encoder else "embedding_dim"},
    }

    if is_cross_encoder and "token_type_ids" in getattr(tokenizer, "model_input_names", []):
        token_type_ids = tokenized.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(tokenized["input_ids"])
        args.append(token_type_ids.to(device))
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "seq_length"}
    elif not is_cross_encoder and cfg.dimensions:
        hidden_size = int(getattr(inner.config, "hidden_size"))
        args.append(torch.full((len(args[0]),), hidden_size, dtype=torch.int64, device=device))
        input_names.append("dimensions")
        dynamic_axes["dimensions"] = {0: "batch_size"}

    onnx_path = output_path / "model.onnx"
    export_kwargs = {
        "model": export_model,
        "args": tuple(args),
        "f": str(onnx_path),
        "input_names": input_names,
        "output_names": [output_name],
        "dynamic_axes": dynamic_axes,
        "opset_version": cfg.opset,
    }

    try:
        with torch.no_grad(), _onnx_safe_bidirectional_mask():
            try:
                torch.onnx.export(**export_kwargs, dynamo=False)
            except TypeError:
                # Older torch has no `dynamo` kwarg.
                torch.onnx.export(**export_kwargs)
    except Exception:
        logger.exception("ONNX export failed for %s at %s", model_type.value, model_path)
        raise

    tokenizer_dir = output_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    with torch.no_grad():
        reference = export_model.eval()(*args)
    feed_args = [tensor.detach().cpu() for tensor in args]
    verify_onnx_matches_reference(
        onnx_path=onnx_path,
        feed={name: tensor.numpy() for name, tensor in zip(input_names, feed_args)},
        reference=reference.float().cpu().numpy(),
        atol=1e-2 if cfg.precision == "fp16" else 1e-3,
    )

    logger.info("ONNX %s exported to %s", model_type.value, onnx_path)
    return onnx_path


def verify_onnx_matches_reference(
    onnx_path: Path,
    feed: dict,
    reference,
    atol: float,
    providers: list[str] | None = None,
) -> float:
    """Return max abs diff. Raise ``ValueError`` on shape mismatch or diff > *atol*."""
    import numpy as np
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=providers or ["CUDAExecutionProvider"],
    )
    expected_inputs = {inp.name for inp in session.get_inputs()}
    missing = expected_inputs - feed.keys()
    if missing:
        raise ValueError(f"ONNX graph at {onnx_path} expects inputs not provided: {sorted(missing)}")

    output_names = [out.name for out in session.get_outputs()]
    actual = np.asarray(session.run(output_names, {k: v for k, v in feed.items() if k in expected_inputs})[0])
    expected = np.asarray(reference)

    if actual.shape != expected.shape:
        raise ValueError(f"ONNX output shape {actual.shape} does not match HuggingFace output shape {expected.shape}.")

    max_diff = float(np.max(np.abs(actual.astype("float64") - expected.astype("float64"))))
    if max_diff > atol:
        raise ValueError(
            f"ONNX output diverges from the HuggingFace checkpoint: max abs diff {max_diff:.3e} > atol {atol:.3e}."
        )
    logger.info("ONNX verification passed (max abs diff %.3e <= atol %.3e)", max_diff, atol)
    return max_diff


def _resolve_export_config(customizer_config: TrainingStepConfig) -> ExportConfig:
    """``retrieval.export``, or defaults if unset."""
    retrieval = getattr(customizer_config, "retrieval", None)
    export = getattr(retrieval, "export", None) if retrieval is not None else None
    return export if isinstance(export, ExportConfig) else ExportConfig()


def _restructure_encoder_output(output_path: Path, primary: str) -> None:
    """Put *primary* at the fileset root; nest the other under ``alternates/``.

    After export, HF is already at the root and ONNX is in ``alternates/onnx``.
    ``primary="hf"`` is therefore a no-op; ``primary="onnx"`` swaps them.
    """
    onnx_dir = output_path / "alternates" / "onnx"
    if primary != "onnx":
        logger.info("Encoder output: hf at top level, onnx in %s", onnx_dir.relative_to(output_path))
        return

    hf_dir = output_path / "alternates" / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    for entry in list(output_path.iterdir()):
        if entry.name == "alternates":
            continue
        shutil.move(str(entry), str(hf_dir / entry.name))
    for entry in list(onnx_dir.iterdir()):
        shutil.move(str(entry), str(output_path / entry.name))
    onnx_dir.rmdir()

    logger.info("Encoder output: onnx at top level, hf in %s", hf_dir.relative_to(output_path))


def process_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    customizer_config: TrainingStepConfig,
    model_type: ModelType = ModelType.LLM,
    resolved_chat_template: str | None = None,
) -> CheckpointInfo:
    """
    Process checkpoint to standard output format.

    Works for LLM, embedding (biencoder), and cross-encoder models.

    Handles three scenarios:
    1. Full weights training: Copy checkpoint, fix FSDP2 arch, preserve chat template (LLM only)
    2. LoRA (unmerged): Copy adapter, preserve format as hf-peft
    3. LoRA merged: Merge adapter with base model, output as standard HF

    Args:
        checkpoint_path: Path to the checkpoint directory (model files)
        output_path: Where to write the processed checkpoint
        customizer_config: Training configuration with model paths and settings
        model_type: Type of model ("llm", "embedding", or "cross_encoder")
        resolved_chat_template: Pre-resolved chat template from training config (LLM only).
            If provided, this template is used. Otherwise, falls back to
            priority-based resolution using model.name and model.path.

    Returns:
        CheckpointInfo with output path, format, and precision
    """
    output_path.mkdir(parents=True, exist_ok=True)

    finetuning_type = customizer_config.training.finetuning_type
    base_model_path = customizer_config.model.path
    job_trust_remote_code = customizer_config.model.trust_remote_code
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
        # Full-weight merge is required for ONNX export and embedding/ranking NIM serving.
        if is_embedding:
            merge_lora_embedding_adapter(
                adapter_path=checkpoint_path,
                base_model_path=base_model_path,
                output_path=output_path,
                trust_remote_code=job_trust_remote_code,
            )
        elif is_cross_encoder:
            merge_lora_cross_encoder_adapter(
                adapter_path=checkpoint_path,
                base_model_path=base_model_path,
                output_path=output_path,
                trust_remote_code=job_trust_remote_code,
            )
        else:
            merge_lora_adapter(
                adapter_path=checkpoint_path,
                base_model_path=base_model_path,
                output_path=output_path,
                trust_remote_code=job_trust_remote_code,
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
        # Full weights: copy Automodel's consolidated dir, then (for untrusted
        # encoders) strip training-class metadata after ONNX export.
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

    if (is_embedding or is_cross_encoder) and checkpoint_format != CheckpointFormat.HF_PEFT:
        export_cfg = _resolve_export_config(customizer_config)
        # Full-SFT copy still has Automodel auto_map until sanitize runs.
        load_trust_remote_code = True if finetuning_type == FinetuningType.ALL_WEIGHTS else job_trust_remote_code
        export_onnx(
            model_path=output_path,
            output_path=output_path / "alternates" / "onnx",
            tokenizer_path=base_model_path,
            model_type=model_type,
            cfg=export_cfg,
            trust_remote_code=job_trust_remote_code,
            load_trust_remote_code=load_trust_remote_code,
        )
        if not job_trust_remote_code:
            sanitize_encoder_hf(output_path, Path(base_model_path))
        _restructure_encoder_output(output_path, export_cfg.primary)

    # Determine precision: use explicit config value, or extract from base model
    precision = customizer_config.model.precision
    if precision is None:
        precision = extract_precision_from_model_config(customizer_config.model.path)

    return CheckpointInfo(
        path=str(output_path),
        format=checkpoint_format,
        precision=precision,
    )


def _move_contents(source: Path, destination: Path) -> None:
    """Move everything in *source* into *destination* and drop the empty *source*."""
    destination.mkdir(parents=True, exist_ok=True)
    for entry in list(source.iterdir()):
        shutil.move(str(entry), str(destination / entry.name))
    source.rmdir()


def copy_checkpoint_stats(workspace_dir: Path, output_path: Path) -> None:
    """Copy per-checkpoint validation statistics into the output fileset."""
    source = workspace_dir / "checkpoints" / CHECKPOINT_STATS_FILENAME
    if not source.is_file():
        logger.warning("Checkpoint statistics were not found at %s", source)
        return
    shutil.copy2(source, output_path / CHECKPOINT_STATS_FILENAME)


def process_selected_checkpoints(
    checkpoints: Mapping[str, Path],
    output_path: Path,
    workspace_dir: Path,
    customizer_config: TrainingStepConfig,
    model_type: ModelType = ModelType.LLM,
    resolved_chat_template: str | None = None,
) -> CheckpointInfo:
    """Process the selected checkpoint(s) into one fileset.

    Every checkpoint is staged into ``{output_path}/{label}``, so ONNX
    conversion and sanitizing happen identically for each. The published
    checkpoint is then moved up to the root, and a ``last`` that was not the
    published one moves under ``alternates/``.
    """
    infos: dict[str, CheckpointInfo] = {}
    for label, checkpoint_path in checkpoints.items():
        infos[label] = process_checkpoint(
            checkpoint_path,
            output_path / label,
            customizer_config,
            model_type=model_type,
            resolved_chat_template=resolved_chat_template,
        )

    root_label = "best" if "best" in infos else "last"
    logger.info("Publishing the %s checkpoint at the fileset root", root_label)
    _move_contents(output_path / root_label, output_path)

    staged_last = output_path / "last"
    if staged_last.exists():
        alternates_path = output_path / "alternates"
        alternates_path.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_last), str(alternates_path / "last"))
        logger.info("Published the last checkpoint under alternates/last")

    copy_checkpoint_stats(workspace_dir, output_path)
    return infos[root_label].model_copy(update={"path": str(output_path)})
