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
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from nhx.automodel.tasks.training.model_utils.constants import ADAPTER_FILES


class TargetCheckpointType(str, Enum):
    """Target checkpoint format types for model conversion."""

    NEMO = "NEMO"
    HF = "HF"
    HF_LORA = "HF_LORA"


logger = logging.getLogger(__name__)


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
    except OSError:
        logger.exception("Cannot read directory '%s'", path)
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


def is_nemo_model_directory(model_path: Path | str) -> bool:
    nemo_structure = {
        "context": {"nemo_tokenizer": {}, "model.yaml": None},
        "weights": {"metadata.json": None},
    }
    return check_directory_structure(model_path, nemo_structure)


def is_huggingface_model_directory(model_path: Path | str) -> bool:
    """
    Checks if a directory contains the necessary files to be considered a
    Hugging Face model directory.

    Args:
        directory_path: The path to the directory to check.

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

    # 2. Check for the presence of model weight files (either safetensors or pytorch bin)
    safe_tensor_file = model_path / "model.safetensors"
    has_safetensors = safe_tensor_file.is_file() or any(model_path.glob("model-*.safetensors"))
    if has_safetensors:
        return True

    logger.debug(f"Missing model weights files in the form of {safe_tensor_file} or {model_path}/model-*.safetensors")
    pytorch_bin_file = model_path / "pytorch_model.bin"
    has_pytorch_bin = pytorch_bin_file.is_file() or any(model_path.glob("pytorch_model-*.bin"))
    if has_pytorch_bin:
        return True

    logger.debug(f"Missing model weights files in the form of {pytorch_bin_file} or {model_path}/pytorch_model-*.bin")
    return False


def determine_llm_model_type(model_dir: str | Path) -> TargetCheckpointType | None:
    """
    Determines whether a model directory contains a HuggingFace or NVIDIA NeMo model.
    """
    model_path = Path(model_dir).resolve()

    if not model_path.exists() or not model_path.is_dir():
        logger.error(f"Provided path {model_path} is not a directory")
        return None

    logger.debug(f"Checking model in {model_path} for LoRA adapter format indicators")
    if is_adapter_file_present(get_flat_files_list(str(model_path))):
        logger.info(f"Huggingface LoRA adapter format detected in {model_path}")
        return TargetCheckpointType.HF_LORA

    logger.debug(f"Checking model in {model_path} for NeMo format indicators")
    if is_nemo_model_directory(model_path):
        logger.info(f"NeMo format detected in {model_path}")
        return TargetCheckpointType.NEMO

    logger.debug(f"Checking model in {model_path} for HugginFace format indicators")
    if is_huggingface_model_directory(model_path):
        logger.info(f"HuggingFace format detected in {model_path}")
        return TargetCheckpointType.HF

    logger.warning(f"model at {model_path} is an unknown checkpoint format")
    logger.warning(f"File List: {get_flat_files_list(str(model_path))}")

    return None
