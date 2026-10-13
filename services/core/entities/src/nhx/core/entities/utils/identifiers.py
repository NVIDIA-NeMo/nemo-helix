# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for generating entity identifiers and names."""

import random
import re
import string
import uuid

import base58


def _normalize_entity_type(entity_type: str) -> str:
    """Normalize entity type for use in IDs and names.

    Converts underscores to dashes and removes other non-alphanumeric characters.

    Args:
        entity_type: The entity type (e.g., 'customization_config')

    Returns:
        Normalized type prefix (e.g., 'customization-config')
    """
    # Convert underscores to dashes
    normalized = entity_type.replace("_", "-")
    # Remove any non-alphanumeric characters except dashes
    normalized = re.sub(r"[^a-zA-Z0-9-]", "", normalized)
    return normalized.lower()


def generate_random_suffix(length: int = 5) -> str:
    """Generate a random suffix for entity names.

    Args:
        length: Number of characters (default 5)

    Returns:
        Random lowercase alphanumeric string
    """
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def generate_entity_name(entity_type: str) -> str:
    """Generate a short, human-readable entity name.

    Format: {entity_type_normalized}-{random_suffix}

    Args:
        entity_type: The entity type (e.g., 'customization_config')

    Returns:
        A generated name like 'customization-config-4qfjo'

    Examples:
        >>> generate_entity_name('customization_config')
        'customization-config-4qfjo'
        >>> generate_entity_name('job')
        'job-7hxkp'
        >>> generate_entity_name('evaluation_result')
        'evaluation-result-2mnvw'
    """
    type_prefix = _normalize_entity_type(entity_type)
    suffix = generate_random_suffix(5)
    return f"{type_prefix}-{suffix}"


def generate_entity_id(entity_type: str) -> str:
    """Generate a compound entity ID with normalized type prefix and base58-encoded UUID.

    Args:
        entity_type: The entity type (e.g., 'customization_config')

    Returns:
        A compound ID like 'customization-config-5Q2LoF8z8M9JZxZsHwJKNn'

    Examples:
        >>> generate_entity_id('customization_config')
        'customization-config-5Q2LoF8z8M9JZxZsHwJKNn'
        >>> generate_entity_id('job')
        'job-7hxkp9Q2LoF8z8M9JZxZsHw'
    """
    type_prefix = _normalize_entity_type(entity_type)
    u = uuid.uuid4()
    return f"{type_prefix}-{base58.b58encode(u.bytes).decode('ascii')}"


def parse_entity_id(entity_id: str) -> tuple[str, str]:
    """Parse a compound entity ID into its components.

    Args:
        entity_id: The compound ID (e.g., 'customization-config-5Q2LoF8z8M9JZxZsHwJKNn')

    Returns:
        Tuple of (entity_type_prefix, unique_suffix)

    Raises:
        ValueError: If the ID format is invalid
    """
    # Find the last hyphen that separates the base58 UUID suffix
    # The suffix is a base58-encoded UUID (22 chars typically)
    parts = entity_id.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid entity ID format: {entity_id}")
    return parts[0], parts[1]
