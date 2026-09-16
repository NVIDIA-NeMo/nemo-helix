# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from langchain_core.language_models.llms import LLM


class FlexibleLLMBase(LLM):
    """
    A flexible base class for LLM implementation, allowing dynamic initialization
    and configuration of subclasses.

    This class extends the `LLM` class from LangChain, providing:
    - A Pydantic configuration that permits arbitrary types and accepts
      additional attributes not explicitly defined in the model schema.
    - An initializer that dynamically sets attributes based on provided
      keyword arguments or retains existing values.
    - Langchain's LLM has a Config class that allows for arbitrary types but not extra attributes.

    Attributes:
        Config (class): Pydantic configuration class with the following options:
            - arbitrary_types_allowed (bool): Allows inclusion of arbitrary
              user-defined types as fields.
            - extra (str): Permits acceptance and storage of extra attributes
              not defined in the model schema.
    """

    class Config:
        """Pydantic configuration for FlexibleLLMBase."""

        arbitrary_types_allowed = True
        # Langchain's LLM has a Config class that allows for arbitrary types but not extra attributes
        extra = "allow"

    def __init__(self, **kwargs):
        """
        Initialize the FlexibleLLMBase instance with dynamic attributes.

        Args:
            **kwargs: Arbitrary keyword arguments corresponding to the
                      attributes to be set on the instance.

        The initializer sets attributes specified in the class annotations
        based on the provided keyword arguments. If an attribute is not
        present in `kwargs`, it retains its existing value or defaults to
        None if unset.
        """
        super().__init__(**kwargs)
        for name in self.__annotations__:
            setattr(self, name, kwargs.get(name, getattr(self, name, None)))
