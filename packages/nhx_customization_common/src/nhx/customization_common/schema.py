# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Namespaced Pydantic base for customization-backend schemas.

Every backend (automodel, unsloth, …) is merged into one FastAPI app under
``/apis/customization``, so two backends that each define e.g. ``TrainingSpec``
collide in the generated OpenAPI. Pydantic freezes a model's JSON-schema name at
class-creation time, so the prefix has to be applied there — this metaclass does
it. ``class TrainingSpec(AutomodelSchema)`` is emitted as ``AutomodelTrainingSpec``.

Usage: declare a per-backend subclass that sets ``__schema_namespace__`` and
inherit *that* from every model the backend owns::

    class AutomodelSchema(NamespacedModel):
        __schema_namespace__ = "Automodel"

    class TrainingSpec(AutomodelSchema):  # emitted as ``AutomodelTrainingSpec``
        ...

A model whose class name already starts with the prefix (e.g.
``AutomodelJobInput``) is left unchanged, so top-level request/response names stay
stable.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

# ``type(BaseModel)`` is pydantic's ``ModelMetaclass``; deriving from it this way
# avoids importing pydantic internals.
_ModelMeta = type(BaseModel)


class _NamespacedMeta(_ModelMeta):
    def __new__(mcs, name, bases, ns, **kw):
        prefix = ns.get("__schema_namespace__") or next(
            (p for b in bases if (p := getattr(b, "__schema_namespace__", None))), None
        )
        if prefix and not name.startswith(prefix):  # don't double-prefix e.g. AutomodelJobInput
            name = f"{prefix}{name}"
            ns["__qualname__"] = name
        return super().__new__(mcs, name, bases, ns, **kw)


class NamespacedModel(BaseModel, metaclass=_NamespacedMeta):
    """Base for backend schemas.

    Declare a per-backend subclass that sets ``__schema_namespace__`` and inherit
    *that* from every model, so each backend's schemas emit distinct component
    names in the merged ``/apis/customization`` OpenAPI spec.
    """

    # Dunder ClassVar → pydantic ignores it as a field; the metaclass reads it at
    # class-creation time to compute the emitted schema name.
    __schema_namespace__: ClassVar[str | None] = None
    model_config = ConfigDict(extra="forbid")  # the config every backend model already sets
