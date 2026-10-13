# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Shared dataset utilities for training backends.

This module provides schema detection, sample counting, and training schedule
utilities that are shared across all training backends (automodel, megatron_bridge, nemo_rl).
"""

import json
import logging
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from nhx.rl.app.constants import DEFAULT_SEED

logger = logging.getLogger(__name__)

# Dataset directory constants for merged files (we control this structure)
MERGED_DIR = "merged"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "validation.jsonl"

# Heuristic patterns for discovering training files. JSONL only: everything
# downstream (schema detection, _merge_files, _create_val_split) is line-delimited,
# so a plain .json array/object would produce invalid merged data or split failures.
TRAIN_PATTERNS = [
    "train*.jsonl",
    "training*.jsonl",
]
TRAIN_DIRS = ["train", "training"]

# Heuristic patterns for discovering validation files (JSONL only — see above).
VAL_PATTERNS = [
    "val*.jsonl",
    "validation*.jsonl",
    "dev*.jsonl",
]
VAL_DIRS = ["val", "validation", "dev"]


class DatasetSchema(str, Enum):
    """Detected dataset schema type."""

    CHAT = "chat"  # OpenAI messages format: {"messages": [...]}
    SFT = "sft"  # Prompt/completion: {"prompt": ..., "completion": ...}
    CUSTOM = "custom"  # Custom columns via prompt_template
    EMBEDDING = "embedding"  # Retrieval format: {"query": ..., "pos_doc": ..., "neg_doc": [...]}


class DatasetFormatError(Exception):
    """Raised when dataset format is invalid or unsupported."""

    pass


def detect_dataset_schema(
    file_path: Path,
    prompt_template: str | None = None,
) -> tuple[DatasetSchema, tuple[str, ...] | None]:
    """
    Detect dataset schema by sampling the first line.

    Supports four formats:
    1. Chat format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    2. Embedding format: {"query": "...", "pos_doc": "...", "neg_doc": ["...", "..."]}
    3. SFT format: {"prompt": "...", "completion": "..."}
    4. Custom format: Any two-column format specified via prompt_template like "{input} {output}"

    Args:
        file_path: Path to the JSONL dataset file.
        prompt_template: Optional template string with two placeholders like "{input} {output}".

    Returns:
        Tuple of (schema_type, column_keys) where:
        - CHAT: column_keys is None
        - EMBEDDING: column_keys is ("query", "pos_doc", "neg_doc")
        - SFT/CUSTOM: column_keys is (question_col, answer_col)

    Raises:
        DatasetFormatError: If the dataset format cannot be detected or is invalid.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        line = f.readline()

    try:
        obj: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError as e:
        raise DatasetFormatError(f"Invalid JSON in {file_path}: {e}")

    # Check for chat format (OpenAI messages)
    if "messages" in obj and isinstance(obj["messages"], list):
        if len(obj["messages"]) > 0 and isinstance(obj["messages"][0], dict):
            if "role" in obj["messages"][0]:
                logger.info(f"Detected chat dataset format in {file_path}")
                return DatasetSchema.CHAT, None

    # Check for embedding/retrieval format
    # Format: {"query": "...", "pos_doc": "...", "neg_doc": ["...", "..."]}
    if "query" in obj and "pos_doc" in obj and "neg_doc" in obj:
        if isinstance(obj["query"], str) and isinstance(obj["pos_doc"], str) and isinstance(obj["neg_doc"], list):
            logger.info(f"Detected embedding/retrieval dataset format in {file_path}")
            return DatasetSchema.EMBEDDING, ("query", "pos_doc", "neg_doc")

    # Check for custom prompt_template format
    if prompt_template:
        keys = re.findall(r"\{(.*?)\}", prompt_template)
        if len(keys) == 2:
            # Validate keys exist in data
            if all(k in obj for k in keys):
                logger.info(f"Detected custom template format with keys {keys}")
                return DatasetSchema.CUSTOM, (keys[0], keys[1])
            else:
                raise DatasetFormatError(
                    f"prompt_template keys {keys} not found in dataset. Available keys: {list(obj.keys())}"
                )
        else:
            raise DatasetFormatError(f"prompt_template must have exactly 2 placeholders, got: {prompt_template}")

    # Check for standard SFT format (prompt/completion)
    if "prompt" in obj and "completion" in obj:
        logger.info(f"Detected SFT (prompt/completion) format in {file_path}")
        return DatasetSchema.SFT, ("prompt", "completion")

    # Fallback - try to find any two string columns
    string_cols = [k for k, v in obj.items() if isinstance(v, str)]
    if len(string_cols) >= 2:
        logger.warning(f"Could not detect standard format, using first two string columns: {string_cols[:2]}")
        return DatasetSchema.SFT, (string_cols[0], string_cols[1])

    raise DatasetFormatError(
        f"Could not detect dataset format. Expected 'messages' (chat) or "
        f"'prompt'/'completion' (SFT) columns. Found: {list(obj.keys())}"
    )


def _count_jsonl_samples_python(file_path: Path) -> int:
    """Pure Python implementation of line counting (fallback)."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Non-empty line
                count += 1
    return count


def count_jsonl_samples(file_path: Path) -> int:
    """
    Count the number of non-empty lines in a JSONL file.

    Uses grep for efficiency with large files when available,
    falls back to pure Python implementation otherwise.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        Number of non-empty lines (samples) in the file.
    """
    # Check if grep is available
    if shutil.which("grep") is None:
        return _count_jsonl_samples_python(file_path)

    try:
        # Use `grep -c "\S"` to count non-empty lines (excludes trailing empty lines)
        result = subprocess.check_output(["grep", "-c", r"\S", str(file_path)], text=True)
        return int(result.strip())
    except subprocess.CalledProcessError as e:
        # grep exits 1 when there are no matching (non-empty) lines → genuinely empty.
        # Any other exit code (e.g. 2 = file unreadable / regex error) is a real
        # failure, so fall back to the Python counter rather than reporting 0.
        if e.returncode == 1:
            return 0
        return _count_jsonl_samples_python(file_path)
    except OSError:
        # Fallback if subprocess fails for any reason
        return _count_jsonl_samples_python(file_path)


def compute_val_check_interval(
    steps_per_epoch: int,
    max_steps: int,
    val_check_interval: Optional[Union[int, float]] = None,
) -> int:
    """
    Compute how often to run validation (in steps).

    This handles the semantic difference between:
    - float <= 1.0: Fraction of epoch (e.g., 0.5 = validate at 50% of each epoch)
    - int or float > 1.0: Absolute step count

    Args:
        steps_per_epoch: Number of gradient steps per epoch.
        max_steps: Maximum training steps.
        val_check_interval: User-provided interval (float for fraction, int for steps).

    Returns:
        Integer step count for validation interval.

    Raises:
        ValueError: If val_check_interval is negative.
    """
    effective_steps = min(steps_per_epoch, max_steps)

    if val_check_interval is None or val_check_interval == 0:
        # Default: validate once per epoch (or at end if max_steps < steps_per_epoch)
        return effective_steps

    if val_check_interval < 0:
        raise ValueError("val_check_interval cannot be negative")

    # Float <= 1.0: interpret as fraction of epoch
    if isinstance(val_check_interval, float) and val_check_interval <= 1.0:
        interval = max(1, int(val_check_interval * steps_per_epoch))
    else:
        # Integer or float > 1.0: treat as absolute step count
        interval = int(val_check_interval)

    # Cap at effective_steps
    interval = min(interval, effective_steps)

    # Ensure validation happens at least once before training ends
    if interval >= max_steps:
        interval = max(1, max_steps - 1)

    return interval


@dataclass
class PreparedDataset:
    """Result of dataset preparation."""

    merged_dir: Path
    train_file: Path
    validation_file: Path
    train_samples: int
    validation_samples: int


def _discover_files_by_patterns(base_path: Path, patterns: list[str], dirs: list[str]) -> list[Path]:
    """
    Discover files matching patterns or in specific directories.

    Searches for:
    1. Files matching glob patterns in base_path
    2. All .jsonl/.json files in specified subdirectories

    Args:
        base_path: Root directory to search.
        patterns: Glob patterns to match (e.g., ["train*.jsonl"]).
        dirs: Subdirectory names to search (e.g., ["train", "training"]).

    Returns:
        Sorted list of discovered file paths.
    """
    files: set[Path] = set()

    # Pattern matching in base directory
    for pattern in patterns:
        for match in base_path.glob(pattern):
            if match.is_file():
                files.add(match.resolve())

    # Files in subdirectories
    for dir_name in dirs:
        subdir = base_path / dir_name
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.is_file() and f.suffix.lower() in (".jsonl", ".json"):
                    files.add(f.resolve())

    return sorted(files)  # Sorted for deterministic ordering


def discover_dataset_files(dataset_path: Path) -> tuple[list[Path], list[Path]]:
    """
    Discover training and validation files using heuristics.

    Heuristics applied (in order):
    1. Files matching train*/training* patterns → training
    2. Files in train/ or training/ directories → training
    3. Files matching val*/validation*/dev* patterns → validation
    4. Files in val/, validation/, or dev/ directories → validation
    5. If only one .jsonl file found → treat as training (will auto-split)

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        Tuple of (training_files, validation_files).

    Raises:
        DatasetFormatError: If no training files can be found.
    """
    dataset_path = Path(dataset_path).resolve()

    if not dataset_path.exists():
        raise DatasetFormatError(f"Dataset path does not exist: {dataset_path}")

    # If path is a file, treat it as the training file
    if dataset_path.is_file():
        logger.info(f"Dataset path is a file, treating as training data: {dataset_path}")
        return [dataset_path], []

    # Discover training files
    train_files = _discover_files_by_patterns(dataset_path, TRAIN_PATTERNS, TRAIN_DIRS)

    # Discover validation files
    val_files = _discover_files_by_patterns(dataset_path, VAL_PATTERNS, VAL_DIRS)

    # Fallback: if no files found with patterns, check for any .jsonl files
    if not train_files and not val_files:
        all_jsonl = sorted(f for f in dataset_path.glob("*.jsonl") if f.is_file())
        if len(all_jsonl) == 1:
            logger.info(f"Found single JSONL file, treating as training data: {all_jsonl[0]}")
            train_files = all_jsonl
        elif len(all_jsonl) > 1:
            # Ambiguous - could be train/val or multiple training files
            logger.warning(
                f"Found {len(all_jsonl)} JSONL files without clear train/val naming. "
                f"Treating all as training data: {[f.name for f in all_jsonl]}"
            )
            train_files = all_jsonl

    if not train_files:
        raise DatasetFormatError(
            f"No training files found in {dataset_path}. "
            f"Expected files matching patterns like train*.jsonl or a train/ directory."
        )

    logger.info(f"Discovered {len(train_files)} training file(s): {[f.name for f in train_files]}")
    if val_files:
        logger.info(f"Discovered {len(val_files)} validation file(s): {[f.name for f in val_files]}")
    else:
        logger.info("No validation files found - will auto-split from training data")

    return train_files, val_files


def _merge_files(files: list[Path], output_file: Path) -> int:
    """
    Merge multiple JSONL files into a single file.

    Args:
        files: List of files to merge.
        output_file: Output file path.

    Returns:
        Total number of samples (non-empty lines) in merged file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Stream each file in fixed-size chunks rather than reading it whole: dataset
    # shards can be multiple GB and inp.read() would pull an entire shard into
    # memory. We track whether the last chunk ended in a newline so a file that
    # lacks a trailing newline doesn't get concatenated onto the next one.
    chunk_size = 1024 * 1024  # 1 MiB
    with open(output_file, "w", encoding="utf-8") as out:
        for f in files:
            ended_with_newline = True  # empty file → no separator needed
            with open(f, "r", encoding="utf-8") as inp:
                for chunk in iter(lambda inp=inp: inp.read(chunk_size), ""):
                    out.write(chunk)
                    ended_with_newline = chunk.endswith("\n")
            # Ensure a newline separates this file's last record from the next file.
            if not ended_with_newline:
                out.write("\n")

    return count_jsonl_samples(output_file)


def _create_val_split(
    train_file: Path,
    output_train: Path,
    output_val: Path,
    val_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> tuple[int, int]:
    """
    Split a training file into train and validation sets.

    Args:
        train_file: Source training file.
        output_train: Output path for training split.
        output_val: Output path for validation split.
        val_ratio: Fraction of data to use for validation (default: 10%).
        seed: Random seed for reproducible splits (default: 1111).

    Returns:
        Tuple of (train_samples, validation_samples).
    """
    # First pass: count non-empty rows without materializing the file (the merged
    # input can be multi-GB, so we never read it all into memory).
    total = count_jsonl_samples(train_file)

    # A split needs at least one sample on each side. Fail fast for files too small
    # to split rather than emitting an empty train (or validation) set that crashes
    # downstream config compilation.
    if total < 2:
        raise DatasetFormatError(
            f"Training file {train_file} has {total} sample(s); at least 2 are required to "
            "create a train/validation split. Provide more training data or supply a separate "
            "validation file."
        )

    # Clamp to total - 1 so at least one training sample always remains, even when
    # the requested ratio would otherwise consume the whole (tiny) dataset.
    val_size = max(1, int(total * val_ratio))
    val_size = min(val_size, total - 1)

    # Pick which row indices go to validation. A local RNG keeps the split
    # deterministic (important for multi-node) without mutating the process-wide
    # random state that later training code may rely on. Holding only the chosen
    # indices (not the rows) keeps memory bounded.
    rng = random.Random(seed)
    val_indices = set(rng.sample(range(total), val_size))

    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_val.parent.mkdir(parents=True, exist_ok=True)

    # The auto-split-from-merged path passes the same path for both train_file and
    # output_train. Opening output_train for write would truncate the file we still
    # need to stream from `src`, yielding empty splits and destroying the merged
    # train file. When they collide, write the train split to a temp file and
    # atomically replace at the end.
    rewrite_train_in_place = train_file.resolve() == output_train.resolve()
    train_write_path = output_train.with_name(f"{output_train.name}.tmp") if rewrite_train_in_place else output_train

    # Second pass: stream each non-empty row to train/val by index — re-serializing
    # to normalize JSON — without ever holding the whole dataset in memory.
    train_count = 0
    val_count = 0
    with (
        open(train_file, "r", encoding="utf-8") as src,
        open(train_write_path, "w", encoding="utf-8") as train_out,
        open(output_val, "w", encoding="utf-8") as val_out,
    ):
        idx = 0
        for raw in src:
            if not raw.strip():
                continue
            normalized = json.dumps(json.loads(raw)) + "\n"
            if idx in val_indices:
                val_out.write(normalized)
                val_count += 1
            else:
                train_out.write(normalized)
                train_count += 1
            idx += 1

    if rewrite_train_in_place:
        train_write_path.replace(output_train)

    logger.info(
        f"Created validation split: {train_count} train samples, {val_count} val samples ({val_ratio:.0%} split)"
    )

    return train_count, val_count


def prepare_dataset(
    dataset_path: Path,
    output_dir: Optional[Path] = None,
    val_split_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> PreparedDataset:
    """
    Prepare dataset for training by discovering, merging, and optionally splitting files.

    This function:
    1. Discovers training and validation files using heuristics
    2. Merges multiple files into single train.jsonl and val.jsonl
    3. Auto-creates validation split if no validation files found
    4. Returns paths to the prepared files

    Args:
        dataset_path: Path to the dataset directory or file.
        output_dir: Directory for merged output (default: dataset_path/merged).
        val_split_ratio: Fraction for auto-split if no validation data (default: 0.1).
        seed: Random seed for reproducible validation splits (default: 1111).

    Returns:
        PreparedDataset with paths to merged files and sample counts.

    Raises:
        DatasetFormatError: If dataset cannot be prepared.
    """
    dataset_path = Path(dataset_path).resolve()

    # Determine output directory
    if output_dir is None:
        if dataset_path.is_file():
            merged_dir = dataset_path.parent / MERGED_DIR
        else:
            merged_dir = dataset_path / MERGED_DIR
    else:
        merged_dir = Path(output_dir).resolve()

    train_output = merged_dir / TRAIN_FILE
    validation_output = merged_dir / VAL_FILE

    # Discover files
    train_files, val_files = discover_dataset_files(dataset_path)

    # Merge training files
    if len(train_files) == 1 and not val_files:
        # Single file, no validation - need to split
        logger.info("Single training file with no validation data - creating split")
        train_samples, validation_samples = _create_val_split(
            train_files[0],
            train_output,
            validation_output,
            val_ratio=val_split_ratio,
            seed=seed,
        )
    else:
        # Merge training files
        train_samples = _merge_files(train_files, train_output)
        logger.info(f"Merged {len(train_files)} training file(s) → {train_output} ({train_samples} samples)")

        if val_files:
            # Merge validation files
            validation_samples = _merge_files(val_files, validation_output)
            logger.info(
                f"Merged {len(val_files)} validation file(s) → {validation_output} ({validation_samples} samples)"
            )
        else:
            # Auto-split from merged training file
            logger.info("No validation files - creating split from merged training data")
            # Read merged, split, re-write
            train_samples, validation_samples = _create_val_split(
                train_output,
                train_output,
                validation_output,
                val_ratio=val_split_ratio,
                seed=seed,
            )

    return PreparedDataset(
        merged_dir=merged_dir,
        train_file=train_output,
        validation_file=validation_output,
        train_samples=train_samples,
        validation_samples=validation_samples,
    )
