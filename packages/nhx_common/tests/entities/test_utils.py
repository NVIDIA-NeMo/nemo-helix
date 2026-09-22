# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for entity utility functions."""

import pytest
from nhx.common.entities.utils import (
    ModelEntityId,
    ParsedEntityRef,
    parse_adapters_suffix,
    parse_entity_ref,
    parse_model_entity_ref,
)


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


# --- ModelEntityId ------------------------------------------------------------


def test_model_entity_id_plain_qualified():
    """A plain ``workspace/name`` id parses with no adapter fields and is not LoRA."""
    mid = ModelEntityId.parse("prod-ws/llama-3")

    assert mid == ModelEntityId(workspace="prod-ws", base_name="llama-3")
    assert mid.is_lora is False
    assert mid.adapter_workspace is None
    assert mid.adapter_name is None
    assert mid.base_id == "prod-ws/llama-3"
    assert mid.to_composite() == "prod-ws/llama-3"
    assert str(mid) == "prod-ws/llama-3"


def test_model_entity_id_plain_bare_with_default_workspace():
    """A bare name uses the default workspace and stays non-LoRA."""
    mid = ModelEntityId.parse("llama-3", default_workspace="default")

    assert mid == ModelEntityId(workspace="default", base_name="llama-3")
    assert mid.is_lora is False


def test_model_entity_id_lora_composite():
    """A LoRA composite decomposes into base + adapter parts and reports is_lora."""
    mid = ModelEntityId.parse("base-ws/base&adapters/adapter-ws/adapter")

    assert mid == ModelEntityId(
        workspace="base-ws",
        base_name="base",
        adapter_workspace="adapter-ws",
        adapter_name="adapter",
    )
    assert mid.is_lora is True
    assert mid.base_id == "base-ws/base"
    assert mid.to_composite() == "base-ws/base&adapters/adapter-ws/adapter"
    assert str(mid) == "base-ws/base&adapters/adapter-ws/adapter"


def test_model_entity_id_lora_round_trips():
    """parse -> to_composite round-trips both shapes exactly."""
    for raw in ("prod-ws/llama-3", "base-ws/base&adapters/adapter-ws/adapter"):
        assert ModelEntityId.parse(raw).to_composite() == raw


def test_model_entity_id_construct_and_format_lora():
    """Constructing a LoRA id from parts formats to the canonical composite string."""
    mid = ModelEntityId(
        workspace="base-ws",
        base_name="base",
        adapter_workspace="a-ws",
        adapter_name="a-name",
    )

    assert mid.is_lora is True
    assert mid.to_composite() == "base-ws/base&adapters/a-ws/a-name"


@pytest.mark.parametrize(
    "identifier",
    [
        "ws/base&adapters/adapter-ws",  # adapter_part has no '/'
        "ws/base&adapters/",  # empty adapter_part
        "ws/base&adapters//adapter",  # empty adapter_workspace
        "ws/base&adapters/adapter-ws/",  # empty adapter_name
        "ws/&adapters/a-ws/a-name",  # empty base_name
    ],
)
def test_model_entity_id_malformed_composite_raises(identifier: str):
    """An ``&adapters/`` infix that is not a well-formed composite is rejected."""
    with pytest.raises(ValueError, match="invalid LoRA composite model entity id"):
        ModelEntityId.parse(identifier)


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
def test_model_entity_id_invalid_ref_raises(identifier: str):
    """Empty / empty-segment ids propagate the parse_model_entity_ref ValueError."""
    with pytest.raises(ValueError, match="invalid model entity reference|must not be empty"):
        ModelEntityId.parse(identifier, default_workspace="default")


def test_model_entity_id_bare_name_without_default_raises():
    """A bare name without a default workspace is rejected (propagated)."""
    with pytest.raises(ValueError, match="workspace"):
        ModelEntityId.parse("llama-3")


# --- parse_adapters_suffix ----------------------------------------------------


def test_parse_adapters_suffix_well_formed():
    """A well-formed composite name decomposes into (base, adapter_ws, adapter_name)."""
    assert parse_adapters_suffix("base&adapters/a-ws/a-name") == ("base", "a-ws", "a-name")


def test_parse_adapters_suffix_plain_name_returns_none():
    """A plain (non-composite) name has no adapter suffix."""
    assert parse_adapters_suffix("llama-3") is None


@pytest.mark.parametrize(
    "name",
    [
        "base&adapters/a-ws",  # no '/' in adapter_part
        "base&adapters/",  # empty adapter_part
        "base&adapters//a-name",  # empty adapter_workspace
        "base&adapters/a-ws/",  # empty adapter_name
        "&adapters/a-ws/a-name",  # empty base
    ],
)
def test_parse_adapters_suffix_malformed_returns_none(name: str):
    """A malformed ``&adapters/`` name is not a valid composite (None, not raise)."""
    assert parse_adapters_suffix(name) is None


def test_parse_adapters_suffix_rejects_surplus_segment():
    """A surplus '/' in the adapter tail is not a valid single-segment adapter name."""
    assert parse_adapters_suffix("base&adapters/ws/name/extra") is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adapter_workspace": "a-ws"},  # workspace without name
        {"adapter_name": "a-name"},  # name without workspace
    ],
)
def test_model_entity_id_rejects_partial_adapter_state(kwargs: dict[str, str]):
    """Constructing with only one adapter field set is rejected (all-or-nothing)."""
    with pytest.raises(ValueError, match="provided together"):
        ModelEntityId(workspace="ws", base_name="base", **kwargs)


def test_model_entity_id_rejects_empty_adapter_fields():
    """Empty-string adapter fields are rejected."""
    with pytest.raises(ValueError, match="must not be empty"):
        ModelEntityId(workspace="ws", base_name="base", adapter_workspace="", adapter_name="")
