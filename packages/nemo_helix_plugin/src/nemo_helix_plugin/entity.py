# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin entity interface — base classes for plugin entity definitions.

Plugin authors import from here rather than ``nhx.common`` directly:

    from nemo_helix_plugin.entity import NemoEntity

See :class:`NemoEntity` for usage examples.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

from nemo_helix_plugin.entities import EntityBase


class NemoEntity(EntityBase):
    """Base class for plugin entity definitions.

    Extends :class:`~nhx.common.entities.client.EntityBase` so subclasses work
    transparently with :class:`~nemo_helix_plugin.entity_client.NemoEntitiesClient` for all
    CRUD operations.

    Every concrete subclass must declare its entity type via the ``entity_type``
    class keyword.  Auto-derivation from the class name (the default ``EntityBase``
    behaviour) is not permitted — it produces surprising strings for acronym-heavy
    names (e.g. ``BLEUMetric`` → ``"b_l_e_u_metric"``) and makes the stored entity
    type invisible at the definition site.

    Abstract intermediate base classes (those with ``__abstract__ = True``,
    ``ABC`` as a direct base, or any ``@abstractmethod`` members) are exempt
    from the check.

    Declare a concrete entity class with the ``entity_type`` keyword::

        from nemo_helix_plugin.entity import NemoEntity

        class MyEntity(NemoEntity, entity_type="my_entity"):
            status: str = "pending"

    Use ``__abstract__ = True`` or inherit from ``ABC`` on shared base classes
    that should not have their own entity type::

        class MyBaseEntity(NemoEntity):
            __abstract__ = True
            status: str = "pending"

        from abc import ABC

        class MyBaseEntity(NemoEntity, ABC):
            status: str = "pending"
    """

    __abstract__: ClassVar[bool] = False

    def __init_subclass__(cls, *, entity_type: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if entity_type is not None:
            cls.__entity_type__ = entity_type
            return

        if cls.__dict__.get("__abstract__", False):
            return
        if any(getattr(v, "__isabstractmethod__", False) for v in cls.__dict__.values()):
            return
        if abc.ABC in cls.__bases__:
            return

        raise TypeError(
            f"{cls.__qualname__} must pass entity_type='...' as a class keyword. "
            f"Example: class {cls.__name__}(NemoEntity, entity_type='my_type'): ... "
            f"To define an abstract intermediate base without an entity type, "
            f"set __abstract__ = True on the class or inherit from ABC."
        )
