# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
import os
import re
from typing import Any, Callable, Optional

from jsonschema import exceptions
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from nmp.automodel.entities.values import FinetuningType, TrainingType
from nmp.automodel.tasks.training.datasets.preparation import DatasetFormatError
from nmp.automodel.tasks.training.datasets.schemas import SFTDatasetSchemaType

logger = logging.getLogger(__name__)

# jsonschema.validate() recompiles the schema on every call. Retrieval JSONL is
# large enough that that path takes tens of minutes; compile once and reuse.
_VALIDATE_PROGRESS_INTERVAL = 10_000


def _compile_validator(schema: dict[str, Any]) -> Validator:
    cls = validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def SFT_SCHEMA(prompt_template: str | None = None):
    """Generate JSON schema for SFT datasets.

    Uses the SFTDatasetSchemaType union which supports:
    - SFTPromptTemplateDatasetItemSchema: Flexible prompt template format
    - SFTPChatDatasetItemSchema: Chat format with messages and tools

    Args:
        prompt_template: Optional template string with placeholders like "{input} {output}".
                        If None or empty string, defaults to standard prompt/completion format.
                        Ignored for chat format detection.

    Returns:
        JSON schema dict with required fields based on the format.
    """
    from pydantic import TypeAdapter

    # Determine required fields for prompt template format
    if prompt_template is not None and prompt_template != "":
        # Extract placeholders from template
        found_keys = re.findall(r"{(.*?)}", prompt_template)

        # TODO: Are we constrained by len == 2?
        # Check for duplicates
        if len(found_keys) != len(set(found_keys)):
            duplicates = [key for key in found_keys if found_keys.count(key) > 1]
            unique_duplicates = list(dict.fromkeys(duplicates))
            raise ValueError(
                f"Prompt template contains duplicate placeholders: {unique_duplicates}. "
                f"Each placeholder should appear only once."
            )

        prompt_template_keys = found_keys
    else:
        prompt_template_keys = ["prompt", "completion"]

    # Create TypeAdapter for the SFT union type to generate base JSON schema
    adapter = TypeAdapter(SFTDatasetSchemaType)
    schema = adapter.json_schema()

    # Add JSON schema metadata
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "SFT Schema"

    # Update the prompt template sub-schema with required fields from prompt_template_keys
    # The schema structure has $defs with the actual schemas, and oneOf/anyOf with $ref pointers
    if "$defs" in schema:
        # Update the SFTPromptTemplateDatasetItemSchema in $defs
        if "SFTPromptTemplateDatasetItemSchema" in schema["$defs"]:
            template_schema = schema["$defs"]["SFTPromptTemplateDatasetItemSchema"]
            # Add template fields as required properties
            if "properties" not in template_schema:
                template_schema["properties"] = {}
            for key in prompt_template_keys:
                template_schema["properties"][key] = {"type": "string"}
            template_schema["required"] = prompt_template_keys
            template_schema["additionalProperties"] = True
    return schema


SCHEMAS: dict[str, Callable[[str | None], dict]] = {
    TrainingType.SFT.value: SFT_SCHEMA,
    TrainingType.DISTILLATION.value: SFT_SCHEMA,
}


class DatasetValidator:
    """Validator for training datasets.

    This class encapsulates dataset validation logic and avoids parameter drilling
    by storing configuration as instance attributes.

    Example usage after prepare_dataset():
        ```python
        from nmp.automodel.tasks.training.datasets.preparation import prepare_dataset
        from nmp.automodel.tasks.training.datasets.validation import DatasetValidator

        # Prepare datasets
        prepared = prepare_dataset(
            dataset_path=Path(customizer_config.dataset.path),
            output_dir=workspace_dir / "dataset",
        )

        # Validate the prepared datasets
        validator = DatasetValidator(
            training_type=customizer_config.training.training_type,
            finetuning_type=customizer_config.training.finetuning_type,
            prompt_template=customizer_config.dataset.prompt_template,
        )
        validator.validate_dataset(str(prepared.train_file))
        validator.validate_dataset(str(prepared.validation_file))
        ```
    """

    def __init__(
        self,
        training_type: TrainingType,
        finetuning_type: Optional[FinetuningType] = None,
        prompt_template: str | None = None,
    ):
        """Initialize validator with training configuration.

        Args:
            training_type: The type of training (SFT, distillation, etc.)
            finetuning_type: Optional finetuning type (LoRA, all_weights, etc.)
            prompt_template: Optional prompt template for SFT datasets
        """
        self.training_type = training_type
        self.finetuning_type = finetuning_type
        self.prompt_template = prompt_template

    def _validate_json_object(self, obj: dict, validator: Validator) -> None:
        """Validate a JSON object against a compiled JSON Schema validator.

        Args:
            obj: The JSON object to validate
            validator: Compiled jsonschema validator for the dataset schema

        Raises:
            TypeError: If validation fails
        """
        try:
            validator.validate(obj)
        except exceptions.ValidationError as e:
            logger.debug(f"Dataset Schema Validation failed: {str(e)}")
            raise TypeError(f"Dataset Schema Validation failed: {e.message}")
        except Exception as e:
            logger.debug(f"Dataset Schema Validation failed: {str(e)}")
            raise TypeError(f"Dataset Schema Validation failed: {e}")

    def detect_dataset_schema(self, file_path: str) -> str:
        """Detect the dataset schema from the first line of the file.

        Args:
            file_path: Path to the dataset file

        Returns:
            Schema name (e.g., 'sft', 'dpo', 'chat')

        Raises:
            DatasetFormatError: If file format is invalid or doesn't match any schema
        """
        with open(file_path, "r", encoding="utf-8") as f:
            line = f.readline()

        try:
            obj: dict[str, Any] = json.loads(line)
        except Exception as e:
            logger.debug(f"{file_path} has entry which is not valid json. Error: {e}\n{line}")
            raise DatasetFormatError(f"{file_path} has entry which is not a valid json: {e}")

        for schema_name, schema_factory in SCHEMAS.items():
            try:
                validator = _compile_validator(schema_factory(self.prompt_template))
                self._validate_json_object(obj, validator)
            except Exception as e:
                logger.debug(f"Parsed jsonl line does not conform to schema {schema_name}. Error: {e}")
            else:
                logger.debug(f"Parsed jsonl line conforms to schema {schema_name}.")
                return schema_name

        raise DatasetFormatError("Dataset does not match any supported format")

    def validate_dataset(self, file_path: str, dataset_type: Optional[str] = None) -> None:
        """Validate a single dataset file.

        Args:
            file_path: Path to the dataset file
            dataset_type: Optional dataset type to validate against. If None, uses training type from config

        Raises:
            DatasetFormatError: If dataset is empty or validation fails
        """
        # Use provided dataset_type or fall back to training type from config
        if dataset_type is None:
            dataset_type = self.training_type.value

        schema_factory = SCHEMAS.get(dataset_type)
        if not schema_factory:
            # Skip validation for unsupported types
            return

        if os.path.getsize(file_path) == 0:
            raise DatasetFormatError(f"{file_path} is empty")

        validator = _compile_validator(schema_factory(self.prompt_template))
        validated_rows = 0

        # Validate each line in the JSONL file
        with open(file_path, "r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj: dict[str, Any] = json.loads(line)
                except Exception as e:
                    logger.debug(f"{file_path} has entry which is not valid json. Error: {e}\n{line}")
                    raise DatasetFormatError(f"{file_path} has entry which is not valid json: {e}")

                try:
                    self._validate_json_object(obj, validator)
                except Exception as e:
                    logger.debug(f"{file_path} does not conform to the expected schema: {e}")
                    raise DatasetFormatError(f"{file_path} does not conform to the expected schema: {e}")
                validated_rows += 1
                if validated_rows % _VALIDATE_PROGRESS_INTERVAL == 0:
                    logger.info("Validated %s rows in %s", validated_rows, file_path)

        logger.info("Validated %s rows in %s", validated_rows, file_path)


# Backward compatibility: provide standalone functions that create a validator instance
def detect_dataset_schema(
    file_path: str,
    training_type: TrainingType,
    finetuning_type: Optional[FinetuningType] = None,
    prompt_template: str | None = None,
) -> str:
    """Detect the dataset schema from the first line of the file.

    Args:
        file_path: Path to the dataset file
        training_type: The type of training (SFT, DPO, etc.)
        finetuning_type: Optional finetuning type (LoRA, all_weights, etc.)
        prompt_template: Optional prompt template for SFT datasets

    Returns:
        Schema name (e.g., 'sft', 'dpo', 'chat')
    """
    validator = DatasetValidator(training_type, finetuning_type, prompt_template=prompt_template)
    return validator.detect_dataset_schema(file_path)


def validate_dataset(
    file_path: str,
    training_type: TrainingType,
    dataset_type: Optional[str] = None,
    finetuning_type: Optional[FinetuningType] = None,
    prompt_template: str | None = None,
) -> None:
    """Validate a single dataset file.

    Args:
        file_path: Path to the dataset file
        dataset_type: Dataset type to validate against
        training_type: The type of training (SFT, DPO, etc.)
        finetuning_type: Optional finetuning type (LoRA, all_weights, etc.)
        prompt_template: Optional prompt template for SFT datasets
    """
    validator = DatasetValidator(training_type, finetuning_type, prompt_template=prompt_template)
    validator.validate_dataset(file_path, dataset_type)


def validate_datasets(
    file_names: list[str],
    training_type: TrainingType,
    dataset_type: Optional[str] = None,
    finetuning_type: Optional[FinetuningType] = None,
    prompt_template: str | None = None,
) -> None:
    """Validate a list of dataset files.

    Args:
        file_names: List of dataset file paths to validate
        dataset_type: Dataset type to validate against (sft, dpo, embedding)
        training_type: The type of training (SFT, DPO, etc.)
        finetuning_type: Optional finetuning type (LoRA, all_weights, etc.)
        prompt_template: Optional prompt template for SFT datasets (ignored for other dataset types)
    """
    validator = DatasetValidator(training_type, finetuning_type, prompt_template=prompt_template)
    for file_name in file_names:
        validator.validate_dataset(file_name, dataset_type)
