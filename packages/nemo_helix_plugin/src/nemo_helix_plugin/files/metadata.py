# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata types for filesets.

The metadata uses a tagged/keyed structure where the key indicates the type:
    metadata: {dataset: {schema: {...}}}

The key in metadata should match the fileset's purpose field.
"""

from __future__ import annotations

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetMetadataContent(BaseModel):
    """Content for dataset-type filesets."""

    # Use `schema_` because `schema` is a BaseModel method.
    model_config = ConfigDict(serialize_by_alias=True)

    schema_: dict | str | None = Field(
        default=None,
        alias="schema",
        description="Default row schema for files in this fileset, either inline JSON Schema or a schema_defs key.",
    )
    schema_defs: dict[str, dict] = Field(
        default_factory=dict,
        description="Reusable JSON Schema definitions keyed by name for deduplicating per-file dataset schemas.",
    )
    schemas_by_path: dict[str, dict | str] = Field(
        default_factory=dict,
        description=(
            "Optional per-file row schemas keyed by relative path within the fileset. "
            "Each value may be inline JSON Schema or a schema_defs key."
        ),
    )

    @model_validator(mode="after")
    def validate_schema_refs(self) -> "DatasetMetadataContent":
        for ref_name, ref_value in [("schema", self.schema_), *self.schemas_by_path.items()]:
            if isinstance(ref_value, str) and ref_value not in self.schema_defs:
                raise ValueError(f"dataset metadata reference '{ref_name}' points to unknown schema_def '{ref_value}'")
        return self

    @model_validator(mode="after")
    def validate_json_schemas(self) -> "DatasetMetadataContent":
        def _validate_schema_document(schema: dict, ref_name: str) -> None:
            validator = validator_for(schema)
            try:
                validator.check_schema(schema)
            except SchemaError as e:
                raise ValueError(
                    f"dataset metadata field '{ref_name}' contains invalid JSON Schema: {e.message}"
                ) from e

        if isinstance(self.schema_, dict):
            _validate_schema_document(self.schema_, "schema")

        for schema_name, schema in self.schema_defs.items():
            _validate_schema_document(schema, f"schema_defs.{schema_name}")

        for path, schema in self.schemas_by_path.items():
            if isinstance(schema, dict):
                _validate_schema_document(schema, f"schemas_by_path.{path}")

        return self


class ToolCallingMetadataContent(BaseModel):
    """Content for tool-calling configuration on model filesets.

    Stores chat template and tool calling settings that are merged into
    the ModelSpec during checkpoint analysis.
    """

    chat_template: str | None = Field(
        default=None,
        description="Jinja2 chat template for the model.",
    )
    tool_call_parser: str | None = Field(
        default=None,
        description="Name of the tool call parser (e.g., 'openai', 'hermes', 'pythonic', 'llama3_json', 'mistral').",
    )
    tool_call_plugin: str | None = Field(
        default=None,
        description="Reference to a fileset containing a custom tool call plugin Python file. "
        "Expected format: '{workspace}/{fileset_name}'.",
    )
    auto_tool_choice: bool | None = Field(
        default=None,
        description="Whether to enable automatic tool choice.",
    )


class ModelMetadataContent(BaseModel):
    """Content for model-type filesets.

    Contains tool calling configuration that is merged into the ModelSpec
    during checkpoint analysis.
    """

    tool_calling: ToolCallingMetadataContent | None = None


class EnvironmentMetadataContent(BaseModel):
    """Content for environment-type filesets (GRPO Gym packages)."""

    model_config = ConfigDict(extra="forbid")

    format: str = Field(description="Environment package format (native-v1, wheels-v1, adapter-wheels-v1).")
    name: str
    adapter_agent: str | None = None
    hub_id: str | None = None
    vf_env_id: str | None = None
    config_paths: list[str] = Field(default_factory=list)


class FilesetMetadata(BaseModel):
    """Tagged metadata container - the key indicates the type.

    Example:
        metadata = FilesetMetadata(
            dataset=DatasetMetadataContent(
                schema={"columns": ["id", "name"]},
            )
        )
    """

    dataset: DatasetMetadataContent | None = None
    model: ModelMetadataContent | None = None
    environment: EnvironmentMetadataContent | None = None

    @model_validator(mode="after")
    def _at_most_one_typed_key(self) -> "FilesetMetadata":
        populated = sum(1 for key in (self.dataset, self.model, self.environment) if key is not None)
        if populated > 1:
            raise ValueError("FilesetMetadata must have at most one typed key populated")
        return self


def environment_metadata_from_manifest(manifest_dict: dict) -> EnvironmentMetadataContent:
    """Build FilesetMetadata.environment from parsed nemo-environment.yaml."""
    meta = manifest_dict.get("metadata") or {}
    adapter = manifest_dict.get("adapter") or {}
    return EnvironmentMetadataContent(
        format=str(manifest_dict["format"]),
        name=str(meta.get("name") or "environment"),
        adapter_agent=adapter.get("agent") or meta.get("adapter_agent"),
        hub_id=meta.get("hub_id"),
        vf_env_id=meta.get("vf_env_id"),
        config_paths=list(manifest_dict.get("config_paths") or []),
    )
