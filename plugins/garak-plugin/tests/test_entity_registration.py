# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the garak_plugin plugin's nemo.entities entry-point factory."""

from __future__ import annotations

import pytest
from garak_plugin.entities import ScanConfig, ScanTarget, get_entity_types
from pydantic import BaseModel, ValidationError


def test_get_entity_types_returns_scan_classes() -> None:
    types = get_entity_types()
    assert ScanConfig in types
    assert ScanTarget in types


def test_every_returned_class_is_a_pydantic_basemodel() -> None:
    for cls in get_entity_types():
        assert issubclass(cls, BaseModel), f"{cls.__qualname__} must be a pydantic BaseModel"


def test_every_returned_class_declares_an_entity_type() -> None:
    seen: set[str] = set()
    for cls in get_entity_types():
        entity_type = getattr(cls, "__entity_type__", None)
        assert isinstance(entity_type, str) and entity_type, (
            f"{cls.__qualname__} must declare a non-empty __entity_type__"
        )
        assert entity_type not in seen, f"duplicate entity_type {entity_type!r} in get_entity_types()"
        seen.add(entity_type)


def test_returned_classes_validate_required_fields() -> None:
    # Roundtrip the entity service's contract: name + workspace + data are
    # all required to construct the model. ScanTarget further requires
    # 'type' and 'model'.
    ScanConfig(name="ok", workspace="default")
    ScanTarget(name="ok", workspace="default", type="nim", model="meta/llama")

    with pytest.raises(ValidationError):
        ScanTarget(name="bad", workspace="default", type="nim")  # missing 'model'
