# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Checkpoint publication for NeMo-RL training.

Copies HuggingFace safetensors (full weights or LoRA adapters) out of a NeMo-RL
step directory. Converts DCP only when ``.metadata`` is present.
"""

import glob
import json
import logging
import os
import re
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Where a LoRA adapter can sit inside a NeMo-RL checkpoint, most specific first.
LORA_ADAPTER_SEARCH_PATHS: tuple[Path, ...] = (
    Path("policy") / "weights" / "model",
    Path("policy") / "weights",
    Path(),
)

# Tokenizer lives beside the weights, not inside them.
RL_TOKENIZER_SUBPATH = Path("policy") / "tokenizer"

# Full-weight HF trees, most specific first.
HF_FULL_WEIGHT_SEARCH_PATHS: tuple[Path, ...] = (
    Path("policy") / "weights" / "model" / "consolidated",
    Path("policy") / "weights" / "model",
    Path("policy") / "weights",
)

_DCP_METADATA_SEARCH_PATHS: tuple[Path, ...] = (
    Path("policy") / "weights" / "model",
    Path("policy") / "weights",
)

# Files that must not appear in the published HuggingFace root.
_HF_SKIP_NAMES = {
    ".hf_metadata",
    "consolidated",
    "consolidate.sh",
    "fqn_to_file_index_mapping.json",
    "fqn_to_dtype_mapping.json",
}

# shard-<rank>-model-<i>-of-<n>.safetensors
_AUTOMODEL_SHARD_RE = re.compile(r"^shard-\d+-(model-\d+-of-(\d+)\.safetensors)$")

# What lets from_pretrained resolve a tensor to one of several weight files.
_HF_WEIGHT_INDEX = "model.safetensors.index.json"


def is_peft_publication(*, requested_lora: bool, adapter_root: Path | None) -> bool:
    """True when the published tree is a LoRA adapter rather than full weights."""
    return requested_lora and adapter_root is not None


def find_lora_adapter_root(checkpoint_path: Path) -> Path | None:
    """Return the directory holding ``adapter_config.json``, or None if there is none."""
    for relative in LORA_ADAPTER_SEARCH_PATHS:
        candidate = checkpoint_path / relative
        if (candidate / "adapter_config.json").is_file():
            return candidate
    return None


def copy_lora_adapter(checkpoint_path: Path, adapter_root: Path, output_path: Path) -> None:
    """Copy an adapter tree to ``output_path``, adding the tokenizer when it is elsewhere."""
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(adapter_root, output_path, dirs_exist_ok=True)
    _copy_tokenizer_if_missing(checkpoint_path, output_path)


def _copy_tokenizer_if_missing(checkpoint_path: Path, output_path: Path) -> None:
    if (output_path / "tokenizer_config.json").is_file():
        return
    tokenizer_dir = checkpoint_path / RL_TOKENIZER_SUBPATH
    if not tokenizer_dir.is_dir():
        logger.warning(
            "No tokenizer found at %s; the published tree is without one",
            tokenizer_dir,
        )
        return
    logger.info("Copying tokenizer from %s to %s", tokenizer_dir, output_path)
    shutil.copytree(tokenizer_dir, output_path, dirs_exist_ok=True)


def _is_peft_weight_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() or (path / "adapter_model.safetensors").is_file()


def _has_dcp_metadata(path: Path) -> bool:
    return (path / ".metadata").is_file()


def _weight_safetensors(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.glob("*.safetensors") if p.is_file() and p.name != "adapter_model.safetensors")


def find_hf_full_weight_root(checkpoint_path: Path) -> Path | None:
    """Return the directory holding full-weight HuggingFace safetensors, or None.

    Prefers ``model/consolidated`` when it contains ``config.json``. Skips DCP and PEFT
    directories. An empty ``consolidated/`` is ignored because Automodel creates that
    directory before writing into it.
    """
    for relative in HF_FULL_WEIGHT_SEARCH_PATHS:
        candidate = checkpoint_path / relative
        if _has_dcp_metadata(candidate) or _is_peft_weight_dir(candidate):
            continue
        if relative.name == "consolidated" and not (candidate / "config.json").is_file():
            continue
        if _weight_safetensors(candidate):
            return candidate
    return None


def find_dcp_weights_root(checkpoint_path: Path) -> Path | None:
    """Return the DCP directory that contains ``.metadata``, or None."""
    for relative in _DCP_METADATA_SEARCH_PATHS:
        candidate = checkpoint_path / relative
        if _has_dcp_metadata(candidate):
            return candidate
    return None


def _flatten_hf_metadata(model_dir: Path, output_path: Path) -> None:
    metadata_dir = model_dir / ".hf_metadata"
    if not metadata_dir.is_dir():
        return
    for item in metadata_dir.iterdir():
        if item.name in _HF_SKIP_NAMES:
            continue
        dest = output_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def _promote_automodel_shards(output_path: Path) -> None:
    """Rename a single-rank, single-file Automodel shard to ``model.safetensors``.

    ``shard-<rank>-model-*-of-*`` files are DCP rank slices. Spanning several files is
    only loadable with an index, which Automodel writes under ``consolidated/`` rather
    than beside the shards, so fail when neither is present.
    """
    groups: dict[str, list[Path]] = {}
    totals: set[int] = set()
    for path in _weight_safetensors(output_path):
        match = _AUTOMODEL_SHARD_RE.match(path.name)
        if match:
            groups.setdefault(match.group(1), []).append(path)
            totals.add(int(match.group(2)))
    if not groups:
        return

    if (output_path / _HF_WEIGHT_INDEX).is_file():
        # The index names the files it maps, so renaming would invalidate it.
        logger.info(f"Publishing Automodel shards under {output_path} as indexed by {_HF_WEIGHT_INDEX}")
        return

    one_rank_per_file = all(len(files) == 1 for files in groups.values())
    one_logical_file = len(groups) == 1 and totals == {1}
    if not (one_rank_per_file and one_logical_file):
        n_files = sum(len(files) for files in groups.values())
        raise ValueError(
            f"Cannot publish Automodel rank shards under {output_path}: {n_files} shard files "
            f"across {len(groups)} logical HF file(s), and no {_HF_WEIGHT_INDEX} to resolve "
            "tensors against them. Training must write model/consolidated via save_consolidated."
        )

    (shard,) = next(iter(groups.values()))
    target = output_path / "model.safetensors"
    if target.exists():
        return
    logger.info("Publishing %s as %s", shard.name, target.name)
    shard.rename(target)


def _fix_fsdp2_architecture(model_path: Path) -> None:
    """Strip the FSDP prefix FSDP2 may write into ``config.json`` architectures."""
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s to fix FSDP2 architectures: %s", config_path, exc)
        return
    original = config.get("architectures")
    if not original:
        return
    fixed = [arch.removeprefix("FSDP") if isinstance(arch, str) else arch for arch in original]
    if original == fixed:
        return
    config["architectures"] = fixed
    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Fixed FSDP2 architecture names: %s -> %s", original, fixed)


def copy_hf_full_weights(checkpoint_path: Path, weights_root: Path, output_path: Path) -> None:
    """Copy a HuggingFace full-weight tree to ``output_path``.

    Flattens ``.hf_metadata`` onto the output root and does not copy optimizer state.
    Raises ``ValueError`` if the tree is Automodel rank shards without a consolidated export.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    for item in weights_root.iterdir():
        if item.name in _HF_SKIP_NAMES:
            continue
        dest = output_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    _flatten_hf_metadata(weights_root, output_path)
    if weights_root.name == "consolidated":
        _flatten_hf_metadata(weights_root.parent, output_path)
    _promote_automodel_shards(output_path)
    _copy_tokenizer_if_missing(checkpoint_path, output_path)
    _fix_fsdp2_architecture(output_path)


def convert_dcp_to_huggingface(
    dcp_checkpoint_path: Path,
    output_path: Path,
) -> Path:
    """Convert a DCP checkpoint to HuggingFace format.

    Args:
        dcp_checkpoint_path: Path to the DCP checkpoint directory
        output_path: Path for the output HuggingFace checkpoint
        model_config: Optional model configuration overrides

    Returns:
        Path to the converted HuggingFace checkpoint
    """
    # Imported here, not at module scope: nemo_rl and transformers exist only in the
    # training image, and the adapter helpers in this module must stay importable (and
    # unit-testable) from the platform environment, which ships neither.
    from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf
    from transformers import AutoModelForCausalLM

    with open(dcp_checkpoint_path / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_name_or_path = config["policy"]["model_name"]
    tokenizer_name_or_path = f"{dcp_checkpoint_path}/policy/tokenizer"

    # It saves the weights as a single pytorch_model.bin file (pickle-based PyTorch format).
    hf_ckpt = convert_dcp_to_hf(
        dcp_ckpt_path=f"{dcp_checkpoint_path}/policy/weights",
        hf_ckpt_path=str(output_path),
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        overwrite=True,
    )

    saved_hf_checkpoint_path = Path(hf_ckpt)
    if not saved_hf_checkpoint_path.exists():
        raise FileNotFoundError(
            f"HF checkpoint not found at {saved_hf_checkpoint_path} after conversion from DCP to HF"
        )
    # Compare resolved paths: convert_dcp_to_hf() may return an absolute path while
    # output_path is relative, and string inequality would then falsely trip even
    # when both point at the same directory.
    if output_path.resolve() != saved_hf_checkpoint_path.resolve():
        raise ValueError(
            f"Output path {output_path} does not match the saved HF checkpoint path {saved_hf_checkpoint_path}"
        )

    # Convert pickle-based .bin format to safetensors format
    # Shards the model into multiple files if larger than 4GB
    model = AutoModelForCausalLM.from_pretrained(saved_hf_checkpoint_path)
    model.save_pretrained(
        saved_hf_checkpoint_path,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    # Remove unnecessary files from DCP checkpoint
    # *.bin files come from the DCP format, which is not needed in the HF safetensors format
    for f in glob.glob(os.path.join(saved_hf_checkpoint_path, "*.bin")) + glob.glob(
        os.path.join(saved_hf_checkpoint_path, "*.bin.index.json")
    ):
        os.remove(f)

    logger.info("Saved HF checkpoint successfully")

    return saved_hf_checkpoint_path
