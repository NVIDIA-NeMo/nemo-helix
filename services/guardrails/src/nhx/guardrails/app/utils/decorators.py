# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from pydantic import BaseModel


def inject_pydantic_fields(pydantic_model: Type[BaseModel]):
    """Decorator to add fields from a specified Pydantic model to a class dynamically.

    Args:
        pydantic_model (Type[BaseModel]): Pydantic model from which to extract fields.

    Returns:
        Type: The decorated class with added fields.
    """

    # TODO: use defaults mixin to exclude None and not supported fileds
    def decorator(cls: Type) -> Type:
        # check if annotations exist, if not initialize
        if not hasattr(cls, "__annotations__"):
            cls.__annotations__ = {}

        # iterate over the fields defined in the pydantic model
        for field_name, field_def in pydantic_model.model_fields.items():
            default = field_def.default
            default_factory = field_def.default_factory

            # Check if a default value is provided, otherwise use the default factory
            if default is not None:
                setattr(cls, field_name, default)
            elif default_factory is not None:  # Ensure default_factory is callable before calling
                setattr(cls, field_name, default_factory())
            else:
                setattr(cls, field_name, None)

            # sett type annotations
            cls.__annotations__[field_name] = field_def.annotation

        return cls

    return decorator
