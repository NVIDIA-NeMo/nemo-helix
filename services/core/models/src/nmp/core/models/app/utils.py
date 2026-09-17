# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for models service."""

import hashlib
import re
from enum import Enum
from logging import getLogger
from typing import Generic, List, Optional, TypeVar

from nemo_platform_plugin.k8s_naming import (
    DNS_LABEL_MAX_LENGTH,
    DNS_SUBDOMAIN_MAX_LENGTH,
    HASH_SUFFIX_LENGTH,
    k8s_safe_name,
    workspace_name_identity,
)
from nemo_platform_plugin.models.types import ModelDeployment, ModelDeploymentConfig, ModelEntity, ModelProvider
from nmp.common.api.common import PaginationData
from nmp.common.entities.constants import NAME_PATTERN as ENTITY_NAME_PATTERN
from pydantic import BaseModel

logger = getLogger(__name__)

T = TypeVar("T")


class PaginatedResult(BaseModel, Generic[T]):
    """Generic container for paginated repository results.

    Used by repository layer to return strongly-typed paginated data.
    The service layer converts this to a full Page response with filter/search/sort metadata.
    """

    data: List[T]
    pagination: PaginationData


class ModelWeightsType(str, Enum):
    """Enum representing the source location of model weights."""

    BAKED_CONTAINER = "baked_container"  # Weights baked into container image
    HUGGINGFACE = "huggingface"  # Weights from HuggingFace Hub
    FILES_SERVICE = "files_service"  # Weights from NeMo Platform Files service
    EXTERNAL_PROVIDER = "external_provider"  # External provider (OpenAI, Anthropic, etc.)
    UNKNOWN = "unknown"  # Unable to determine weights location


class ModelConfigParseError(ValueError):
    """Exception raised when model configuration parsing fails due to invalid input."""

    pass


def parse_model_name_revision(
    model_namespace: Optional[str] = None,
    model_name: Optional[str] = None,
    model_revision: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse model namespace, name, and revision with precedence rules.

    Supports parsing from model_name (namespace/name@revision) or explicit parameters.

    Precedence:
    - Explicit model_namespace/model_revision take precedence over parsed values
    - If model_namespace provided, model_name is NOT parsed for namespace/ prefix
    - Revision remains None if not specified (no default)
    - If no fields provided at all, returns None for all (baked-in weights case)

    Args:
        model_namespace: Explicit namespace (optional) - refers to HuggingFace/NeMo Platform model source
        model_name: Model name, may contain "namespace/" prefix and/or "@revision" suffix (optional)
        model_revision: Explicit revision (optional)

    Returns:
        Tuple of (namespace, name, revision). All can be None.

    Raises:
        ModelConfigParseError: If both model_revision and @revision suffix are specified
    """
    # Case 1: No parameters provided - container has baked-in weights
    if not model_namespace and not model_name and not model_revision:
        return None, None, None

    parsed_namespace = model_namespace
    parsed_name = model_name
    parsed_revision = model_revision

    # Parse model_name if provided
    if model_name:
        # Check for @revision suffix
        name_has_revision = "@" in model_name
        if name_has_revision:
            name_without_revision, suffix_revision = model_name.rsplit("@", 1)

            # Validate: cannot have both explicit revision and @revision suffix
            if model_revision:
                raise ModelConfigParseError(
                    f"Cannot specify both model_revision field ('{model_revision}') and "
                    f"@revision suffix in model_name ('{model_name}'). Please use only one."
                )

            parsed_revision = suffix_revision
            parsed_name = name_without_revision

        # Parse namespace prefix only if explicit model_namespace was NOT provided
        if not model_namespace and parsed_name is not None and "/" in parsed_name:
            # Split on first / to extract namespace
            parts = parsed_name.split("/", 1)
            parsed_namespace = parts[0]
            parsed_name = parts[1]

    return parsed_namespace, parsed_name, parsed_revision


def is_multi_llm_image(image_name: Optional[str]) -> bool:
    # Default is nvcr.io/nim/nvidia/llm-nim but we support other prefixes, as long as the image ends in `llm-nim`
    if not image_name:
        return True
    multi_llm_name = "llm-nim"
    split_image_name = image_name.split("/")
    return split_image_name[-1] == multi_llm_name


def get_model_weights_type(
    model_provider: Optional[ModelProvider] = None,
    model_deployment: Optional[ModelDeployment] = None,
    model_deployment_config: Optional[ModelDeploymentConfig] = None,
    model_entity: Optional[ModelEntity] = None,
) -> ModelWeightsType:
    """Determine the source location of model weights based on deployment configuration.

    This function analyzes the deployment context to determine where model weights are sourced from.
    Used for determining if special handling is needed (e.g., model puller for HuggingFace,
    NIMCache for SFT models) and for populating artifact fields during autodiscovery.

    Args:
        model_provider: Optional ModelProvider - only needed for EXTERNAL_PROVIDER detection
        model_deployment: Optional ModelDeployment - deployment context
        model_deployment_config: Optional ModelDeploymentConfig - checked for image_name and model_name
        model_entity: Optional ModelEntity - checked for SFT full weights and artifact URLs

    Returns:
        ModelWeightsType enum indicating the weights source
    """
    if not model_provider and not model_deployment and not model_deployment_config and not model_entity:
        # There should never be a case where no provider/deployment/config/entity is provided
        logger.warning(
            "No model_provider, model_deployment, model_deployment_config, or model_entity provided, unable to determine weights type"
        )
        return ModelWeightsType.UNKNOWN
    if model_provider and model_provider.model_deployment_id and (not model_deployment or not model_deployment_config):
        # When the provider has a deployment, the relevant deployment/config should also be passed
        logger.warning(
            "ModelProvider has model_deployment_id but deployment/config not provided, unable to determine weights type"
        )
        return ModelWeightsType.UNKNOWN

    # Check simplest cases first
    if model_provider and not model_provider.model_deployment_id:
        return ModelWeightsType.EXTERNAL_PROVIDER

    # Model entity with artifact (fileset) always uses Files service path. The puller
    # uses HF_ENDPOINT to talk to NeMo Platform Files, which resolves the fileset (e.g. to HF Hub).
    if model_entity and model_entity.fileset:
        return ModelWeightsType.FILES_SERVICE

    executor_cfg = model_deployment_config.executor_config if model_deployment_config else None
    model_spec_cfg = model_deployment_config.model_spec if model_deployment_config else None
    image_name = executor_cfg.image_name if executor_cfg else None
    model_name = model_spec_cfg.model_name if model_spec_cfg else None

    # If the model is a multi-LLM, we have already ruled out HF weights, so we download from Files service
    if is_multi_llm_image(image_name) and model_name:
        logger.debug("Detected Files service weights via multi-LLM: downloading from NeMo Platform Files service")
        return ModelWeightsType.FILES_SERVICE
    # Baked container weights are the default assumed case for model-specific NIM images
    if not is_multi_llm_image(image_name):
        logger.debug("Detected baked container weights: model-specific NIM image with model_name")
        return ModelWeightsType.BAKED_CONTAINER

    logger.warning("Unable to determine weights location")
    return ModelWeightsType.UNKNOWN


def normalize_model_entity_name(model_name: str) -> str:
    """Normalize a model name to match the entity store NAME_PATTERN (RFC 1035-style).

    Entity store requires: start with [a-z], length 2-63, only [a-z0-9-] (and
    temporarily @ . + _), no consecutive hyphens, no trailing hyphen. This function
    normalizes when possible; if the result would not match, it raises ValueError
    so callers can skip or fail explicitly.

    Names that would otherwise normalize cleanly but begin with a digit (e.g.
    upstream catalog ids like ``01-ai/yi-large``) are prefixed with ``m-`` so they
    satisfy the leading-letter requirement. The prefix is purely internal to the
    entity store / routing layer; the original id is preserved as the
    ``served_model_name`` and remains the user-facing handle.

    Args:
        model_name: The original model name (e.g., "meta/llama-3.2-1b-instruct")

    Returns:
        Normalized model name valid for entity store (e.g., "meta-llama-3-2-1b-instruct")

    Raises:
        ValueError: If the name cannot be normalized to a valid entity name (e.g. empty,
            only invalid characters, or single character).

    Examples:
        >>> normalize_model_entity_name("meta/llama-3.2-1b-instruct")
        "meta-llama-3-2-1b-instruct"
        >>> normalize_model_entity_name("model:v1.0")
        "model-v1-0"
        >>> normalize_model_entity_name("01-ai/yi-large")
        "m-01-ai-yi-large"
        >>> normalize_model_entity_name("")
        ValueError: ... cannot be normalized to a valid entity name
    """
    pattern = re.compile(ENTITY_NAME_PATTERN)
    # Lowercase and replace non-alphanumeric with hyphens
    normalized = model_name.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    # Collapse consecutive hyphens (entity store forbids --)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized:
        raise ValueError(
            f"Model name {model_name!r} cannot be normalized to a valid entity name: "
            "result is empty (use at least one letter or digit)."
        )
    # Entity store NAME_PATTERN requires a leading [a-z]. Upstream catalogs sometimes
    # publish digit-leading ids (e.g. "01-ai/yi-large"); prefix them with "m-" so they
    # become routable as internal entity names. The prefix is added *before* the 63-char
    # truncation step so the existing truncate+hash machinery still produces a valid
    # length-bounded result for long digit-leading names. Idempotent: names that already
    # start with a letter are left untouched, so re-normalizing "m-01-ai-yi-large" stays
    # "m-01-ai-yi-large".
    if normalized[0].isdigit():
        normalized = f"m-{normalized}"
    # If over 63 chars, truncate with deterministic hash suffix to avoid collisions (before validating)
    if len(normalized) > 63:
        hash_suffix = hashlib.sha256(model_name.encode()).hexdigest()[:8]
        max_base_len = 63 - len(hash_suffix) - 1  # room for '-' + hash
        truncated = normalized[:max_base_len].rstrip("-")
        if not truncated or not truncated[-1].isalnum():
            while truncated and not truncated[-1].isalnum():
                truncated = truncated[:-1]
            if not truncated:
                raise ValueError(
                    f"Model name {model_name!r} cannot be normalized to a valid entity name: "
                    "truncation would leave an invalid name."
                )
        normalized = f"{truncated}-{hash_suffix}"
    if not pattern.match(normalized):
        raise ValueError(
            f"Model name {model_name!r} normalizes to {normalized!r}, which is not valid. "
            "Entity names must start with a lowercase letter, be 2-63 characters, "
            "and contain only lowercase letters, digits, and hyphens (no consecutive hyphens)."
        )
    return normalized


_LORA_SIDECAR_SUFFIX = "-sidecar"
# Primary Docker container names must leave room for an optional LoRA sidecar suffix.
_DOCKER_CONTAINER_NAME_MAX_LENGTH = DNS_LABEL_MAX_LENGTH - len(_LORA_SIDECAR_SUFFIX)

# Backward-compatible aliases for models service call sites and tests.
_HASH_SUFFIX_LENGTH = HASH_SUFFIX_LENGTH
_get_k8s_safe_name = k8s_safe_name
_workspace_name_identity = workspace_name_identity


def get_docker_container_name(workspace: str, name: str) -> str:
    """Docker NIM container name (capped at 55 chars to leave room for ``-sidecar``)."""
    label_name = f"md-{workspace}-{name}"
    return _get_k8s_safe_name(
        label_name,
        max_length=_DOCKER_CONTAINER_NAME_MAX_LENGTH,
        name_type="label",
        hash_input=_workspace_name_identity(workspace, name),
    )


def get_docker_volume_name(workspace: str, name: str) -> str:
    """Docker model-cache volume name for a deployment."""
    label_name = f"nim-cache-{workspace}-{name}"
    return _get_k8s_safe_name(
        label_name,
        name_type="label",
        hash_input=_workspace_name_identity(workspace, name),
    )


def get_docker_puller_container_name(workspace: str, name: str) -> str:
    """Docker SFT/model puller container name."""
    label_name = f"md-puller-{workspace}-{name}"
    return _get_k8s_safe_name(
        label_name,
        name_type="label",
        hash_input=_workspace_name_identity(workspace, name),
    )


def get_docker_plugin_puller_container_name(workspace: str, name: str) -> str:
    """Docker plugin fileset puller container name."""
    label_name = f"md-plugin-{workspace}-{name}"
    return _get_k8s_safe_name(
        label_name,
        name_type="label",
        hash_input=_workspace_name_identity(workspace, name),
    )


def get_deployment_resource_name(workspace: str, name: str) -> str:
    """
    Generate K8s resource name for ModelDeployment resources (NIMService/PVC).

    This is used for NIMService and standalone PVC resources which must follow
    RFC 1035 DNS label rules (63 char limit). The returned name is
    ``md-{workspace}-{name}-{hash8}`` where the hash is derived from
    ``workspace/name``, not the hyphen-joined prefix alone.

    For NIMCache resources, use `get_nimcache_resource_name` instead, which
    reserves 4 characters for the `-job` suffix appended by k8s-nim-operator.

    Args:
        workspace: The deployment workspace
        name: The deployment name

    Returns:
        A K8s-compliant resource name with 'md-' prefix

    Example:
        >>> get_deployment_resource_name("default", "llama-3.2-1b")
        'md-default-llama-3-2-1b-<hash8>'
    """
    base = f"md-{workspace}-{name}"
    identity = _workspace_name_identity(workspace, name)
    return _get_k8s_safe_name(
        base,
        suffix="",
        name_type="label",
        hash_input=identity,
    )


def get_nimcache_resource_name(workspace: str, name: str) -> str:
    """
    Generate K8s resource name specifically for NIMCache resources.

    Uses a 59-character limit instead of the standard 63, to leave room for
    the `-job` suffix (4 chars) that k8s-nim-operator appends to the NIMCache
    name when creating its internal batch Job.  Without this headroom the Job
    name exceeds the 63-char K8s label limit and the NIMCache reconciler fails:

        Job.batch "<nimcache-name>-job" is invalid:
          metadata.labels: Invalid value: "...": must be no more than 63 characters

    The same `_get_k8s_safe_name` logic is used, so names always include an
    8-char hash suffix. Names that exceed the 59-char limit are deterministically
    truncated before the hash — ensuring GET/LIST/DELETE operations on the same
    (workspace, name) pair always resolve to the same NIMCache resource name.

    Args:
        workspace: The deployment workspace
        name: The deployment name

    Returns:
        A K8s-compliant resource name with 'md-' prefix, capped at 59 characters

    Example:
        >>> get_nimcache_resource_name("default", "llama-3.2-1b")
        'md-default-llama-3-2-1b-<hash8>'
    """
    base = f"md-{workspace}-{name}"
    identity = _workspace_name_identity(workspace, name)
    return _get_k8s_safe_name(
        base,
        max_length=59,
        suffix="",
        name_type="label",
        hash_input=identity,
    )


def get_deployment_secret_name(workspace: str, name: str, prefix: str = "md", suffix: str = "") -> str:
    """
    Generate K8s Secret name with configurable prefix and suffix.

    Secrets use RFC 1123 DNS subdomain rules (253 char limit). The hash is
    derived from ``workspace/name``; the caller suffix (e.g. ``-hf-token``)
    is appended after the hash: ``{base}-{hash8}{suffix}``.

    Args:
        workspace: The workspace
        name: The resource name
        prefix: Prefix to prepend (default: "md")
        suffix: Suffix to append (e.g., "-hf-token", "-api-key")

    Returns:
        A K8s-compliant secret name

    Examples:
        >>> get_deployment_secret_name("default", "test", prefix="md", suffix="-hf-token")
        'md-default-test-<hash8>-hf-token'

        >>> get_deployment_secret_name("default", "test", prefix="model-provider", suffix="-api-key")
        'model-provider-default-test-<hash8>-api-key'
    """
    base = f"{prefix}-{workspace}-{name}"
    identity = _workspace_name_identity(workspace, name)
    return _get_k8s_safe_name(
        base,
        max_length=DNS_SUBDOMAIN_MAX_LENGTH,
        suffix=suffix,
        name_type="dns_subdomain",
        hash_input=identity,
    )
