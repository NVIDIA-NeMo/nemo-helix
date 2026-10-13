# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common value objects for the Guardrails service."""

import os
import warnings
from typing import Any, List, Optional, Union

from nhx.common.entities import SYSTEM_WORKSPACE
from nhx.common.entities.values import Value
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic.v1 import validator

from ..enums import RoleEnum
from ._private import (
    ExceptionContent,
    GenerationLog,
    GenerationOptions,
    RailsConfig,
)


class GuardrailsDataInput(Value):
    config: Optional[Union[str, RailsConfig]] = Field(
        default=os.getenv("DEFAULT_CONFIG_ID", f"{SYSTEM_WORKSPACE}/default"),
        description="The id of the configuration or its dict representation to be used.",
    )

    # it uses config if config is a str
    config_id: Optional[str] = Field(
        default=os.getenv("DEFAULT_CONFIG_ID", f"{SYSTEM_WORKSPACE}/default"),
        description="The id of the configuration to be used.",
    )
    config_ids: Optional[List[str]] = Field(
        default=None,
        description="The list of configuration ids to be used. If set, the configurations will be combined.",
        validate_default=True,
    )
    return_choice: bool = Field(
        default=False,
        description="If set, guardrails data will be included as a JSON in the choices array.",
    )
    context: Optional[dict] = Field(
        default=None,
        description="Additional context data to be added to the conversation.",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="If set, partial message deltas will be sent, like in ChatGPT. "
        "Tokens will be sent as data-only server-sent events as they become "
        "available, with the stream terminated by a data: [DONE] message.",
    )
    # let's make it generic and the validation happens on the MS side
    options: GenerationOptions = Field(
        default_factory=GenerationOptions,
        description="Additional options for controlling the generation.",
    )
    state: Optional[dict] = Field(
        default=None,
        description="A state object that should be used to continue the interaction.",
    )

    # Validators ensure that only one of config, config_id, or config_ids is set
    # and appropriately populate related fields based on the input.

    @model_validator(mode="before")
    @classmethod
    def set_config_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("config"):
                if isinstance(data.get("config"), str):
                    # ensure_only_oneof_config_and_config_id
                    if data.get("config_id"):
                        raise ValueError("Only one of config_id or config of type str should be specified")
                    data["config_id"] = data["config"]
                    data["config"] = None
                else:
                    data["config_id"] = None
                    data["config_ids"] = None

        return data

    @model_validator(mode="before")
    @classmethod
    def ensure_config_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("config_id") is not None and data.get("config_ids") is not None:
                raise ValueError("Only one of config_id or config_ids should be specified")
            if data.get("config_id") is None and data.get("config_ids") is not None:
                data["config_id"] = None
            if data.get("config"):
                warnings.warn("No config_id or config_ids provided, using the DataConfig in request body")
        return data

    @field_validator("config_ids", mode="after")
    @classmethod
    def ensure_config_ids(cls, v, info: ValidationInfo):
        if v is None and info.data.get("config_id") and info.data.get("config_ids") is None:
            # Populate config_ids with config_id if only config_id is provided
            return [info.data["config_id"]]
        return v


class GuardrailsDataOutput(Value):
    llm_output: Optional[dict] = Field(default=None, description="Contains any additional output coming from the LLM.")
    config_ids: Optional[List[str]] = Field(default=None, description="The list of configuration ids that were used.")
    output_data: Optional[dict] = Field(
        default=None,
        description="The output data, i.e. a dict with the values corresponding to the `output_vars`.",
    )
    log: Optional[GenerationLog] = Field(default=None, description="Additional logging information.")


class CheckResponseItem(Value):
    role: RoleEnum = Field(..., description="Role of the response entity.")
    content: Union[ExceptionContent, str] = Field(..., description="Content based on the role.")

    @validator("content", pre=True)
    def validate_content(cls, v, values):
        if "role" in values:
            role = values["role"]
            if role == RoleEnum.EXCEPTION and isinstance(v, dict):
                return ExceptionContent(**v)
            elif role == RoleEnum.ASSISTANT and isinstance(v, str):
                return v
            else:
                raise ValueError(
                    f"Invalid content for role '{role}'. Expected {'dict' if role == RoleEnum.EXCEPTION else 'str'}."
                )
        return v
