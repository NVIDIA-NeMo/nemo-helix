# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_helix_plugin.entity — NemoEntity base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest
from nemo_helix_plugin.entities import EntityBase
from nemo_helix_plugin.entity import NemoEntity

# ---------------------------------------------------------------------------
# NemoEntity.__init_subclass__ enforcement
# ---------------------------------------------------------------------------


def test_entity_type_keyword_accepted() -> None:
    """Concrete subclass with entity_type keyword is accepted at definition."""

    class Widget(NemoEntity, entity_type="widget"):
        colour: str = "red"


def test_missing_entity_type_raises() -> None:
    """Concrete subclass without entity_type raises TypeError at class definition."""
    with pytest.raises(TypeError, match="entity_type"):

        class BadEntity(NemoEntity):
            pass


def test_abstract_flag_skips_check() -> None:
    """Subclass with __abstract__ = True is not required to declare entity_type."""

    class AbstractBase(NemoEntity):
        __abstract__ = True
        shared_field: str = ""

    class ConcreteChild(AbstractBase, entity_type="concrete_child"):
        pass


def test_abstract_flag_concrete_child_still_required() -> None:
    """Abstract intermediate base does not satisfy the check for its concrete children."""

    class AbstractBase(NemoEntity):
        __abstract__ = True

    with pytest.raises(TypeError, match="entity_type"):

        class MissingType(AbstractBase):
            pass


def test_abc_base_skips_check() -> None:
    """Subclass with ABC as a direct base is exempt from the entity_type check."""

    class AbstractBase(NemoEntity, ABC):
        shared_field: str = ""

    class ConcreteChild(AbstractBase, entity_type="abc_concrete_child"):
        pass


def test_abc_base_concrete_child_still_required() -> None:
    """ABC intermediate base does not satisfy the check for its concrete children."""

    class AbstractBase(NemoEntity, ABC):
        pass

    with pytest.raises(TypeError, match="entity_type"):

        class MissingType(AbstractBase):
            pass


def test_abstractmethods_skip_check() -> None:
    """Class with unresolved @abstractmethod is exempt (ABC pattern)."""

    class AbstractPlugin(NemoEntity, ABC):
        @abstractmethod
        def run(self) -> None: ...

    class ConcretePlugin(AbstractPlugin, entity_type="concrete_plugin"):
        def run(self) -> None:
            pass


def test_child_without_entity_type_raises() -> None:
    """Child of a concrete entity must pass its own entity_type keyword."""

    class Parent(NemoEntity, entity_type="parent"):
        pass

    with pytest.raises(TypeError, match="entity_type"):

        class Child(Parent):
            extra_field: str = ""


def test_child_with_entity_type_accepted() -> None:
    """Child that passes its own entity_type keyword is accepted."""

    class Parent(NemoEntity, entity_type="parent2"):
        pass

    class Child(Parent, entity_type="child2"):
        extra_field: str = ""


def test_entity_type_sets_dunder() -> None:
    """The entity_type keyword sets __entity_type__ on the class."""

    class Gizmo(NemoEntity, entity_type="gizmo"):
        pass

    assert Gizmo.__entity_type__ == "gizmo"


def test_nemo_entity_is_entity_base_subclass() -> None:
    """NemoEntity subclasses are also EntityBase subclasses (EntityClient compatibility)."""

    class Gadget(NemoEntity, entity_type="gadget"):
        pass

    gadget = Gadget(workspace="default")
    assert isinstance(gadget, EntityBase)
