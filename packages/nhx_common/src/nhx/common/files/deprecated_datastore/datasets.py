# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated Dataset and URN models.

This module provides backward compatibility for the Dataset and URN classes
that were previously located in nhx_common.datamodel.datastore.datasets
and nhx_common.datamodel.types.

New code should use the entity-store API or SDK directly.
"""

from typing import Any, Optional

from pydantic import AnyUrl, BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class URN(str):
    """A URN (Uniform Resource Name) for NeMo Helix resources.

    This is a backward compatibility class that provides proper Pydantic support.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.str_schema()


class Dataset(BaseModel):
    """A dataset that can be used for fine-tuning or evaluation.

    This is a simplified version for backward compatibility.
    It does not extend EntityBase and is not intended for entity store operations.
    """

    model_config = {"protected_namespaces": ()}

    format: Optional[str] = Field(
        default=None,
        description="Specifies the dataset format, referring to the schema of the dataset rather than the file format. Examples include SQuAD, BEIR, etc.",
    )
    files_url: AnyUrl = Field(
        description="The location where the artifact files are stored. This can be a URL pointing to NDS, Hugging Face, S3, or any other accessible resource location."
    )
    hf_endpoint: Optional[AnyUrl] = Field(
        default=None,
        description="For HuggingFace URLs, the endpoint that should be used. By default, this is set to the Data Store URL. "
        'For HuggingFace Hub, this should be set to "https://huggingface.co".',
    )
    split: Optional[str] = Field(
        default=None,
        description="The split of the dataset. Examples include train, validation, test, etc.",
    )
    limit: Optional[int] = Field(
        default=None,
        description="The maximum number of items to be used from the dataset.",
    )

    # Additional fields that may be used in some contexts (from VersionedEntity)
    id: Optional[str] = Field(default=None, description="Entity ID")
    name: Optional[str] = Field(default=None, description="Entity name")
    namespace: Optional[str] = Field(default=None, description="Namespace")
    description: Optional[str] = Field(default=None, description="Description")
