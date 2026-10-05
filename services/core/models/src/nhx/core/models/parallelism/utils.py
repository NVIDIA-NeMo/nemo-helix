# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def divisors(n: int) -> list[int]:
    """Return all divisors of n in ascending order."""
    return [d for d in range(1, n + 1) if n % d == 0]


def compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    """Compile regex patterns for module name matching."""
    return [re.compile(p) for p in patterns]


def name_matches(name: str, include_res, exclude_res):
    """Check if a module name matches include/exclude patterns."""
    if exclude_res and any(r.search(name) for r in exclude_res):
        return False
    if not include_res:
        return True
    return any(r.search(name) for r in include_res)


def get_flat_files_list(parent_dir: str) -> List[str]:
    """
    Get a list of files in a directory
    """
    parent_path = Path(parent_dir).resolve()
    if not parent_path.exists():
        raise ValueError(f"Path {parent_dir} does not exist")
    if not parent_path.is_dir():
        raise ValueError(f"Path {parent_dir} is not a directory")

    return [str(path) for path in parent_path.rglob("*") if path.is_file()]


def is_adapter_file_present(files: List[str]) -> bool:
    """
    Check if the any file is a LoRA adapter file
    """
    for file in files:
        if not file:
            continue
        if any(adapter_file in file.lower() for adapter_file in ADAPTER_FILES):
            return True
    return False


def check_directory_structure(path: Path | str, target: Dict[str, Optional[Dict]]) -> bool:
    if isinstance(path, str):
        path = Path(path)

    if not path.is_dir():
        logger.error(f"Provided path '{path}' is not a directory")
        return False

    try:
        got_files = {f.name for f in path.iterdir()}
    except OSError as e:
        logger.error(f"Cannot read directory '{path}'. Reason: {e}")
        return False

    expected_files = set(target.keys())
    missing = expected_files - got_files
    if missing:
        logger.debug(f"Mismatch in '{path}': Missing items -> {missing}")
        return False

    for name, _target in target.items():
        current_path = path / name
        if isinstance(_target, dict):
            # this is a directory
            if not current_path.is_dir():
                return False
            if not check_directory_structure(current_path, _target):
                return False
        elif _target is None:
            if not current_path.is_file():
                logger.debug(f"Mismatch: '{current_path}' is expected to be a file but is a directory.")
                return False
    return True


def create_tmpdir_with_files(temp_dir: str, paths: List[str], root: Optional[str] = None) -> Path:
    td = Path(temp_dir)
    for p in paths:
        # account for absolute paths
        if root and p[0] == "/":
            p = p[len(root) :]
        if not p:
            continue
        if p[0] == "/":
            p = p[1:]
        file_path = td / Path(p)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
    return td


def is_nemo_model_directory(model_path: Path | str) -> bool:
    nemo_structure = {
        "context": {"nemo_tokenizer": {}, "model.yaml": None},
        "weights": {"metadata.json": None},
    }
    return check_directory_structure(model_path, nemo_structure)


def _has_weight_files_on_disk(model_path: Path) -> bool:
    """Check for model weight files on the local filesystem."""
    safe_tensor_file = model_path / "model.safetensors"
    if safe_tensor_file.is_file() or any(model_path.glob("model-*.safetensors")):
        return True

    logger.debug(f"Missing model weights files in the form of {safe_tensor_file} or {model_path}/model-*.safetensors")

    pytorch_bin_file = model_path / "pytorch_model.bin"
    if pytorch_bin_file.is_file() or any(model_path.glob("pytorch_model-*.bin")):
        return True

    logger.debug(f"Missing model weights files in the form of {pytorch_bin_file} or {model_path}/pytorch_model-*.bin")
    return False


def _has_weight_files_in_listing(file_listing: list[str]) -> bool:
    """Check for model weight files in a remote file listing.

    The listing contains file paths for weight files "model.safetensors" for small models and
    "model-00001-of-00013.safetensors" for large models that may not exist on the
    local filesystem because we explicitly filter the suffixes in method analyze_checkpoint in run.py
    """
    weight_suffixes = (".safetensors", ".bin", ".safetensors.index.json", ".bin.index.json")
    for path in file_listing:
        if path.endswith(weight_suffixes):
            return True

    logger.debug(f"No weight files found in file listing ({len(file_listing)} entries)")
    return False


def is_huggingface_model_directory(
    model_path: Path | str,
    file_listing: list[str] | None = None,
) -> bool:
    """
    Checks if a directory contains the necessary files to be considered a
    Hugging Face model directory.

    Config and tokenizer files are always validated against the local
    filesystem.  Weight files can be validated against either a remote
    file_listing (when only metadata was downloaded locally) or
    the local filesystem (when the full checkpoint is present).

    Args:
        model_path: The path to the local directory to check.
        file_listing: Optional list of file paths (e.g. from a fileset
            API).  When provided, weight-file validation is performed
            against this listing instead of the local filesystem.

    Returns:
        True if the directory contains a config.json file and model weights,
        False otherwise.
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)

    # 1. Check for the mandatory config.json file
    config_file = model_path / "config.json"
    if not config_file.is_file():
        logger.debug(f"Missing {config_file}")
        return False

    tokenizer_files = [
        model_path / "tokenizer.json",
        model_path / "tokenizer_config.json",
        model_path / "vocab.txt",
        model_path / "merges.txt",
    ]
    if not any(tf.is_file() for tf in tokenizer_files):
        logger.debug(f"Missing any tokenizer file: at least one of [{tokenizer_files}] is required")
        return False

    # 2. Check for the presence of model weight files (either safetensors or pytorch bin).
    # We need to add check for both file_listing and on disk because this function can be called from
    # 1) the fileset API in the model-spec task where weight files are intentionally not
    #    downloaded locally.
    # 2) the CLI and other callers that operate on a full checkpoint.
    if file_listing is not None:
        has_weights = _has_weight_files_in_listing(file_listing)
    else:
        has_weights = _has_weight_files_on_disk(model_path)
    if not has_weights:
        logger.info(f"No model weight files found for {model_path} in file listing: {file_listing} or on disk")

    return has_weights
