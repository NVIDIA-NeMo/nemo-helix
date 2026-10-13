# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sequence packing utilities for Automodel training.

Sequence packing combines multiple shorter sequences into a single packed sequence
to improve GPU utilization during training. This module provides:

1. Optimal pack size calculation based on dataset statistics
2. Dataset sequence length estimation via sampling

The algorithm balances packing efficiency with training stability by:
- Calculating a target packing factor from global batch size and GPU count
- Ensuring pack size is at least the max sequence length in the dataset
- Clamping to the model's maximum sequence length

Usage with Automodel:
    The `packed_sequence_size` calculated here should be passed to Automodel's
    config under `packed_sequence.packed_sequence_size`. Automodel automatically
    handles step calculation based on the packed dataset size - no manual
    adjustment of max_steps or global_batch_size is needed.

Reference:
    - NeMo docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/packed_sequence.html
    - Automodel docs: https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/llm/dataset.md#packed-sequence-support-in-nemo-automodel
"""

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

from nhx.automodel.app.constants import DEFAULT_SEED
from nhx.automodel.tasks.training.schemas import TrainingStepConfig

logger = logging.getLogger(__name__)


@dataclass
class PackingEstimate:
    """Statistics from dataset sampling for sequence packing configuration.

    This dataclass holds the results of sampling a dataset to estimate
    sequence length statistics, which are used to calculate optimal
    pack sizes for sequence packing.

    Attributes:
        pack_size: Recommended pack size in tokens for Automodel's
            `packed_sequence.packed_sequence_size` config
        avg_seq_length: Average sequence length in the sampled data
        max_seq_length: Maximum sequence length in the sampled data
        packing_factor: Estimated number of sequences per pack
            (pack_size / avg_seq_length)
        samples_analyzed: Number of samples successfully tokenized
    """

    pack_size: int
    avg_seq_length: int
    max_seq_length: int
    packing_factor: float
    samples_analyzed: int


def _ceil_even(num: int | float) -> int:
    """Round up to the nearest even number.

    NeMo/Automodel prefer even sequence lengths for efficiency with
    tensor parallelism and other optimizations.

    Examples:
        >>> _ceil_even(3)
        4
        >>> _ceil_even(4)
        4
        >>> _ceil_even(5.5)
        6
    """
    return int(math.ceil(num / 2) * 2)


def calculate_target_packing_factor(config: TrainingStepConfig) -> int:
    """Return the target number of samples per packed sequence."""
    parallelism = config.parallelism
    total_gpus = parallelism.num_nodes * parallelism.num_gpus_per_node
    return max(config.batch.global_batch_size // total_gpus, 1)


def calculate_optimal_pack_size(
    config: TrainingStepConfig,
    dataset_avg_seq_length: int | None = None,
    dataset_max_seq_length: int | None = None,
) -> int:
    """
    Calculate optimal pack size for sequence packing.

    This algorithm balances packing efficiency with training stability:
    1. Target packing_factor = global_batch_size / total_gpus
    2. target_pack_size = avg_seq_length * packing_factor (but at least max_seq_length)
    3. Clamp to model's max_seq_length

    The packing factor determines how many sequences fit into one packed sequence.
    A higher packing factor means better GPU utilization but may affect convergence
    if pack sizes become very large.

    If dataset statistics are not provided, uses model's max_seq_length as a
    conservative default (which effectively disables the optimization).

    Args:
        config: Training configuration containing parallelism, batch, and model settings
        dataset_avg_seq_length: Average sequence length in the dataset (after tokenization)
        dataset_max_seq_length: Maximum sequence length in the dataset

    Returns:
        Optimal pack size in tokens

    Example:
        For a setup with:
        - global_batch_size = 32
        - 8 GPUs (num_nodes=1, num_gpus_per_node=8)
        - avg_seq_length = 512
        - max_seq_length = 1024
        - model.max_seq_length = 4096

        Calculation:
        - packing_factor = 32 / 8 = 4
        - target_pack_size = ceil_even(512 * 4) = 2048
        - final = max(2048, 1024) = 2048 (clamped to 4096) = 2048
    """
    model_max_seq = config.model.max_seq_length

    # If no dataset stats provided, use model's max_seq_length (conservative)
    if dataset_avg_seq_length is None or dataset_max_seq_length is None:
        logger.info(f"No dataset statistics provided, using model max_seq_length: {model_max_seq}")
        return model_max_seq

    # Calculate target packing factor (how many sequences can fit in one pack)
    # This keeps the effective batch size close to the original gbs
    target_packing_factor = calculate_target_packing_factor(config)

    # Calculate pack size based on average sequence length
    # Round to nearest even number for efficiency
    target_pack_size = _ceil_even(round(dataset_avg_seq_length * target_packing_factor))

    # Ensure pack size is at least the max sequence length in the dataset
    # (so no sequence gets truncated due to packing)
    target_pack_size = max(target_pack_size, dataset_max_seq_length)

    # Clamp to model's maximum sequence length
    optimal_pack_size = min(target_pack_size, model_max_seq)

    logger.info(
        f"Calculated optimal pack size: {optimal_pack_size} "
        f"(avg_seq={dataset_avg_seq_length}, max_seq={dataset_max_seq_length}, "
        f"packing_factor={target_packing_factor})"
    )

    return optimal_pack_size


def estimate_dataset_sequence_lengths(
    config: TrainingStepConfig,
    train_file: Path | None = None,
    max_samples: int = 1000,
    seed: int = DEFAULT_SEED,
    trust_remote_code: bool = False,
) -> PackingEstimate | None:
    """
    Estimate dataset sequence lengths by sampling and calculate optimal pack size.

    This is a lightweight alternative to full tokenization that uses reservoir
    sampling to randomly select a subset of the dataset for sequence length
    estimation. The sampling is unbiased regardless of dataset ordering.

    The function:
    1. Loads the model's tokenizer
    2. Randomly samples up to `max_samples` examples using reservoir sampling
    3. Tokenizes each example (using apply_chat_template for chat format)
    4. Calculates optimal pack size based on the statistics

    NOTE: Sampling may underestimate max_seq_length for datasets with rare
    long sequences. The pack size calculation accounts for this by clamping
    to the model's max_seq_length.

    Args:
        config: Training configuration with dataset and model paths
        train_file: Path to the prepared training JSONL file.  When provided
            this file is used directly; otherwise falls back to
            ``config.dataset.path / "train.jsonl"``.
        max_samples: Maximum number of samples to analyze (default: 1000)
        seed: Random seed for reproducible sampling (default: 1111)
        trust_remote_code: Whether to trust remote code (default: False)

    Returns:
        PackingEstimate with pack_size and statistics, or None if estimation fails
    """

    try:
        if train_file is None:
            train_file = Path(config.dataset.path) / "train.jsonl"

        if not train_file.exists():
            logger.warning(f"Training file not found: {train_file}")
            return None

        # Import here to avoid ModuleNotFoundError in environments where
        # transformers is not installed (e.g., during test collection)
        from transformers import AutoTokenizer

        # Load tokenizer from model
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.path,
            trust_remote_code=trust_remote_code,
        )

        random.seed(seed)

        # Sample examples to estimate lengths
        lengths = _sample_sequence_lengths(train_file, tokenizer, max_samples)

        if not lengths:
            logger.warning("Could not estimate sequence lengths from dataset")
            return None

        avg_length = _ceil_even(int(sum(lengths) / len(lengths)))
        max_length = _ceil_even(max(lengths))

        # Calculate optimal pack size
        pack_size = calculate_optimal_pack_size(config, avg_length, max_length)
        packing_factor = pack_size / avg_length if avg_length > 0 else 1.0

        estimate = PackingEstimate(
            pack_size=pack_size,
            avg_seq_length=avg_length,
            max_seq_length=max_length,
            packing_factor=round(packing_factor, 2),
            samples_analyzed=len(lengths),
        )

        logger.info(
            f"Packing estimate from {len(lengths)} samples: "
            f"pack_size={pack_size}, avg_seq={avg_length}, max_seq={max_length}, "
            f"packing_factor={estimate.packing_factor:.2f}"
        )

        return estimate

    except Exception as e:
        logger.warning(f"Failed to estimate sequence lengths: {e}")
        return None


def _sample_sequence_lengths(
    train_file: Path,
    tokenizer,
    max_samples: int,
) -> list[int]:
    """
    Sample sequences from a JSONL file and return their tokenized lengths.

    Uses reservoir sampling for unbiased random selection, then tokenizes
    each sample to measure its length. For chat format, uses apply_chat_template
    to get accurate lengths including role tokens and formatting.

    Args:
        train_file: Path to training JSONL file
        tokenizer: HuggingFace tokenizer
        max_samples: Maximum samples to return

    Returns:
        List of sequence lengths (in tokens)
    """
    # Reservoir sampling to select samples
    samples: list[str] = []
    with open(train_file, "r") as f:
        for i, line in enumerate(f):
            if i < max_samples:
                samples.append(line)
            else:
                j = random.randint(0, i)
                if j < max_samples:
                    samples[j] = line

    # Tokenize samples to get lengths
    lengths = []
    for line in samples:
        try:
            obj = json.loads(line)
            length = _get_sample_token_length(obj, tokenizer)
            if length is not None:
                lengths.append(length)
        except Exception:
            # Skip malformed lines
            continue

    return lengths


def _get_sample_token_length(obj: dict, tokenizer) -> int | None:
    """
    Get the tokenized length of a dataset sample.

    For chat format, uses apply_chat_template to accurately measure length
    including role tokens, special tokens, and formatting. Falls back to
    simple text concatenation for other formats or if chat template fails.

    Args:
        obj: Parsed JSON object from dataset
        tokenizer: HuggingFace tokenizer

    Returns:
        Token count, or None if sample is empty/invalid
    """
    # Chat format: use apply_chat_template for accurate length
    if "messages" in obj:
        messages = obj["messages"]
        if messages and hasattr(tokenizer, "apply_chat_template"):
            try:
                tokens = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                )
                return len(tokens)
            except Exception:
                # Fall back to text extraction if chat template fails
                pass

        # Fallback: concatenate role + content
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                if role or content:
                    parts.append(f"{role}: {content}")
        text = "\n".join(parts)
        if text:
            return len(tokenizer.encode(text, add_special_tokens=True))
        return None

    # SFT format: prompt + completion
    if "prompt" in obj and "completion" in obj:
        text = str(obj["prompt"]) + " " + str(obj["completion"])
        return len(tokenizer.encode(text, add_special_tokens=True))

    # Generic: concatenate all string values
    text = " ".join(str(v) for v in obj.values() if isinstance(v, str))
    if text:
        return len(tokenizer.encode(text, add_special_tokens=True))
    return None
