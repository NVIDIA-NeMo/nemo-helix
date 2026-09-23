# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes DNS-safe resource naming shared by platform services and plugins.

Plugins cannot depend on ``nhx_common``; this module lives in ``nemo_helix_plugin``
so deployments, models, and other plugins share one hashing/normalization contract.
"""

import hashlib
import re
from typing import Literal

DNS_LABEL_MAX_LENGTH = 63
DNS_SUBDOMAIN_MAX_LENGTH = 253
HASH_SUFFIX_LENGTH = 8


def workspace_name_identity(workspace: str, name: str) -> str:
    """Canonical ``workspace/name`` identity used as ``hash_input`` for resource names."""
    return f"{workspace}/{name}"


def k8s_safe_name(
    base_name: str,
    max_length: int = DNS_LABEL_MAX_LENGTH,
    suffix: str = "",
    name_type: Literal["label", "dns_subdomain"] = "label",
    hash_input: str | None = None,
    *,
    include_hash: bool = True,
) -> str:
    """Generate a Kubernetes-compliant name with a mandatory hash suffix.

    Normalizes ``base_name`` for display, truncates when needed, then returns
    ``{normalized}-{hash8}{suffix}``. The hash is SHA-256 of ``hash_input`` when
    provided, otherwise ``base_name``. Callers representing a workspace/name pair
    should pass ``hash_input=workspace_name_identity(workspace, name)`` so the hash
    reflects the unambiguous identity rather than a join-ambiguous prefix.
    """
    min_required_length = 1 + len(suffix)
    if include_hash:
        min_required_length += 1 + HASH_SUFFIX_LENGTH
    if min_required_length > max_length:
        raise ValueError("max_length is too small for base name, hash, and suffix")

    hash_source = hash_input if hash_input is not None else base_name
    hash_suffix = hashlib.sha256(hash_source.encode()).hexdigest()[:HASH_SUFFIX_LENGTH]

    normalized = base_name.lower()

    if name_type == "label":
        normalized = re.sub(r"[^a-z0-9-]", "-", normalized)
        normalized = re.sub(r"-+", "-", normalized)
        if normalized and not normalized[0].isalpha():
            normalized = f"x{normalized}"
    else:  # dns_subdomain
        normalized = re.sub(r"[^a-z0-9.-]", "-", normalized)
        normalized = re.sub(r"[.]+", ".", normalized).strip(".")
        labels = []
        for label in normalized.split("."):
            label = re.sub(r"-+", "-", label).strip("-")
            labels.append(label or "x")
        normalized = ".".join(labels)
        normalized = normalized.lstrip("-.")
        if not normalized or not normalized[0].isalnum():
            normalized = f"x{normalized}"

    normalized = normalized.rstrip("-.")
    if not normalized or not normalized[-1].isalnum():
        normalized = "x" if not normalized else normalized.rstrip("-.")
        if not normalized:
            normalized = "x"

    if not include_hash:
        if len(normalized) + len(suffix) <= max_length:
            return f"{normalized}{suffix}"

        max_base_len = max_length - len(suffix)
        if max_base_len < 1:
            max_base_len = 1
        truncated = normalized[:max_base_len].rstrip("-.")
        if not truncated or not truncated[-1].isalnum():
            while truncated and not truncated[-1].isalnum():
                truncated = truncated[:-1]
            if not truncated:
                truncated = "x"
        return f"{truncated}{suffix}"

    reserved = HASH_SUFFIX_LENGTH + 1 + len(suffix)
    if len(normalized) + reserved > max_length:
        max_base_len = max_length - reserved
        if max_base_len < 1:
            max_base_len = max(1, max_length - HASH_SUFFIX_LENGTH - len(suffix))

        truncated = normalized[:max_base_len]
        truncated = truncated.rstrip("-.")
        if not truncated or not truncated[-1].isalnum():
            while truncated and not truncated[-1].isalnum():
                truncated = truncated[:-1]
            if not truncated:
                # Input was all invalid chars, or truncation left no usable base
                # (e.g. max_length only fits hash + suffix).
                truncated = "x"
        normalized = truncated

    result = f"{normalized}-{hash_suffix}{suffix}"

    if len(result) > max_length:
        excess = len(result) - max_length
        normalized = normalized[: len(normalized) - excess].rstrip("-.")
        if not normalized:
            # Rare second-pass trim when hash+suffix still exceed max_length.
            normalized = "x"
        result = f"{normalized}-{hash_suffix}{suffix}"

    return result
