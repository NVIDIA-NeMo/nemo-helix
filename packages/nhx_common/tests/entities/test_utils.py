# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for entity utility functions."""

import pytest
from nhx.common.entities.utils import ParsedEntityRef, parse_entity_ref, parse_model_entity_ref


def test_parse_entity_ref_simple_name_with_default_workspace():
    """Test parsing a simple name with a default workspace."""
    result = parse_entity_ref("my-secret", default_workspace="default")

    assert result == ParsedEntityRef(workspace="default", name="my-secret")


def test_parse_entity_ref_qualified_name():
    """Test parsing a qualified name (workspace/name format)."""
    result = parse_entity_ref("prod-workspace/my-secret")

    assert result == ParsedEntityRef(workspace="prod-workspace", name="my-secret")


def test_ignore_default_workspace_for_qualified_names():
    """Test that qualified names use embedded workspace, not default."""
    result = parse_entity_ref("prod-workspace/my-secret", default_workspace="default")

    assert result == ParsedEntityRef(workspace="prod-workspace", name="my-secret")


def test_parse_entity_ref_simple_name_without_default_raises():
    """Test that simple names without default_workspace raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        parse_entity_ref("my-secret")

    assert "my-secret" in str(exc_info.value)
    assert "workspace" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "identifier",
    [
        "a/b/c",
        "a/b/c/d",
        "/name",
        "workspace/",
        "/",
        "",
    ],
)
def test_parse_entity_ref_invalid_format_raises(identifier: str):
    """Test that malformed references (too many segments or empty segments) raise ValueError."""
    with pytest.raises(ValueError, match="invalid entity reference"):
        parse_entity_ref(identifier, default_workspace="default")


# --- parse_model_entity_ref ---------------------------------------------------


def test_parse_model_entity_ref_simple_name_with_default_workspace():
    """Bare name uses the supplied default workspace."""
    result = parse_model_entity_ref("my-model", default_workspace="default")

    assert result == ParsedEntityRef(workspace="default", name="my-model")


def test_parse_model_entity_ref_qualified_name():
    """Qualified ``workspace/name`` parses straightforwardly."""
    result = parse_model_entity_ref("prod-workspace/my-model")

    assert result == ParsedEntityRef(workspace="prod-workspace", name="my-model")


def test_parse_model_entity_ref_ignores_default_workspace_for_qualified_names():
    """Qualified names use the embedded workspace, not the default."""
    result = parse_model_entity_ref("prod-workspace/my-model", default_workspace="default")

    assert result == ParsedEntityRef(workspace="prod-workspace", name="my-model")


def test_parse_model_entity_ref_qualified_lora_composite():
    """LoRA composite ``base&adapters/adapter_ws/adapter_name`` is preserved as the entity name.

    Splits on the first ``/`` only, matching the cache-key convention used by
    ``ModelCache.rebuild_model_entity_map``.
    """
    result = parse_model_entity_ref("base-ws/base&adapters/adapter-ws/adapter")

    assert result == ParsedEntityRef(workspace="base-ws", name="base&adapters/adapter-ws/adapter")


def test_parse_model_entity_ref_bare_lora_composite_with_default_workspace():
    """Bare LoRA composite (no leading workspace) — first ``/`` separates workspace from name.

    For ``base&adapters/adapter-ws/adapter``, the first ``/`` makes ``base&adapters`` the
    workspace and ``adapter-ws/adapter`` the name. This matches how the model cache is keyed
    when a provider publishes a composite served-model id; the IGW relies on the prefix
    strip in ``openai_proxy`` to produce a name that yields the right cache key.
    """
    result = parse_model_entity_ref("base&adapters/adapter-ws/adapter")

    assert result == ParsedEntityRef(workspace="base&adapters", name="adapter-ws/adapter")


def test_parse_model_entity_ref_simple_name_without_default_raises():
    """Bare names without a default workspace are rejected."""
    with pytest.raises(ValueError) as exc_info:
        parse_model_entity_ref("my-model")

    assert "my-model" in str(exc_info.value)
    assert "workspace" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "identifier",
    [
        "/name",
        "workspace/",
        "/",
        "",
        "   ",
    ],
)
def test_parse_model_entity_ref_invalid_format_raises(identifier: str):
    """Empty segments and empty inputs are rejected even with a default workspace.

    Notably, ``a/b/c`` is **valid** for ``parse_model_entity_ref`` (entity name is
    ``b/c``), unlike ``parse_entity_ref`` which rejects it. That's the whole point
    of this function — preserve composite entity names.
    """
    with pytest.raises(ValueError, match="invalid model entity reference|must not be empty"):
        parse_model_entity_ref(identifier, default_workspace="default")
