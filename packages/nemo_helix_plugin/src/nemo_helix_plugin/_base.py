# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared base for all plugin ABCs — enforces the ``name`` class variable."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar


def _has_unimplemented_abstract(cls: type) -> bool:
    """Return True if *cls* still has any abstract methods it hasn't overridden.

    We cannot rely on ``cls.__abstractmethods__`` here because
    ``__init_subclass__`` fires inside ``type.__new__``, before ``ABCMeta``
    has a chance to compute and set ``__abstractmethods__``.  Instead we walk
    the MRO directly.
    """
    # Methods this class defines itself (may be abstract or concrete)
    if any(getattr(v, "__isabstractmethod__", False) for v in vars(cls).values()):
        # The class itself declares new abstract methods — it's still an ABC.
        return True
    # Inherited abstract methods that are not yet overridden by cls
    for base in cls.__mro__[1:]:
        for attr, val in vars(base).items():
            if getattr(val, "__isabstractmethod__", False):
                cls_attr = getattr(cls, attr, None)
                if getattr(cls_attr, "__isabstractmethod__", False):
                    return True
    return False


class _NamedPlugin(ABC):
    """Mixin that enforces a string ``name`` class variable on concrete subclasses.

    Intermediate ABCs (those that still have unimplemented abstract methods)
    are exempt — only fully concrete subclasses are checked.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if _has_unimplemented_abstract(cls):
            return  # Still an ABC — skip enforcement
        name_val = getattr(cls, "name", None)
        if not isinstance(name_val, str) or not name_val:
            raise TypeError(
                f"{cls.__qualname__} must define a non-empty string class variable 'name'. "
                f'Example:\n\n    class {cls.__name__}(...):\n        name = "my-plugin"'
            )
