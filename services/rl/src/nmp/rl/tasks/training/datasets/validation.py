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
from pathlib import Path
from typing import Any, Callable, Optional

from jsonschema import exceptions
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from nmp.rl.entities.values import FinetuningType, TrainingType
from nmp.rl.schemas.environment import GymDatasetRow
from nmp.rl.tasks.training.datasets.preparation import DatasetFormatError
from nmp.rl.tasks.training.datasets.schemas import (
    DPOPreferenceDatasetSchemaType,
    SFTDatasetSchemaType,
    get_preference_dataset_discriminator,
)

logger = logging.getLogger(__name__)

# jsonschema.validate() recompiles the schema on every call. Compile once and reuse.
_VALIDATE_PROGRESS_INTERVAL = 10_000


def _compile_validator(schema: dict[str, Any]) -> Validator:
    cls = validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def DPO_SCHEMA(_: str | None = None) -> dict:
    """Generate JSON schema for DPO preference datasets.

    Uses the DPOPreferenceDatasetSchemaType union which supports:
    - PreferenceDataset: Native format with context + ranked completions
    - BinaryPreferenceDataset: Simple prompt/chosen/rejected strings
    - HelpSteer3Dataset: NVIDIA HelpSteer3 format with preference scores
    - Tulu3PreferenceDataset: AllenAI Tulu3 format with message lists
    """
    from pydantic import TypeAdapter

    # Create TypeAdapter for the DPO union type to generate JSON schema
    adapter = TypeAdapter(DPOPreferenceDatasetSchemaType)
    schema = adapter.json_schema()

    # Add JSON schema metadata
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

    return schema


def SFT_SCHEMA(prompt_template: str | None = None):
    """Generate JSON schema for SFT datasets.

    Uses the SFTDatasetSchemaType union which supports:
    - SFTPromptTemplateDatasetItemSchema: Flexible prompt template format
    - SFTChatDatasetItemSchema: Chat format with messages and tools

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


def GRPO_SCHEMA(_: str | None = None) -> dict:
    """JSON schema for NeMo Gym GRPO dataset rows.

    Validates the agent-agnostic fields only. Agent-specific keys (``vf_env_id`` for
    verifiers_agent, ``expected_answer`` for the math resources servers) are passed
    through, so a dataset is not rejected for targeting a different Gym agent.
    """
    from pydantic import TypeAdapter

    adapter = TypeAdapter(GymDatasetRow)
    schema = adapter.json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


# The RL backend trains DPO and GRPO; register schemas by TrainingType value.
SCHEMAS: dict[str, Callable[[str | None], dict]] = {
    TrainingType.DPO.value: DPO_SCHEMA,
    TrainingType.GRPO.value: GRPO_SCHEMA,
}


class DatasetValidator:
    """Validator for training datasets.

    This class encapsulates dataset validation logic and avoids parameter drilling
    by storing configuration as instance attributes.

    Example usage from dpo_config.py after prepare_dataset():
        ```python
        from nmp.rl.tasks.training.datasets.preparation import prepare_dataset
        from nmp.rl.tasks.training.datasets.validation import DatasetValidator

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
        *,
        prompt_template: str | None = None,
    ):
        """Initialize validator with training configuration.

        Args:
            training_type: The type of training (DPO, etc.)
            finetuning_type: Optional finetuning type (LoRA, all_weights, etc.)
            prompt_template: Optional prompt template for datasets
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
            Schema name (e.g., 'dpo')

        Raises:
            DatasetFormatError: If file format is invalid or doesn't match any schema
        """
        first_line = _first_nonempty_line(file_path)
        if first_line is None:
            raise DatasetFormatError(f"{file_path} has no non-empty rows")

        try:
            obj: dict[str, Any] = json.loads(first_line)
        except Exception as e:
            # Log identifiers only — the raw row can contain customer training data.
            logger.debug(f"{file_path}: first row is not valid JSON: {e}")
            raise DatasetFormatError(f"{file_path} has an entry which is not valid JSON: {e}")

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
            # Fail loudly: a typo or an unwired training type would otherwise skip
            # validation entirely and let a malformed dataset reach training.
            raise DatasetFormatError(f"Unsupported dataset_type for validation: {dataset_type}")

        if os.path.getsize(file_path) == 0:
            raise DatasetFormatError(f"{file_path} is empty")

        validator = _compile_validator(schema_factory(self.prompt_template))
        is_dpo = dataset_type == TrainingType.DPO.value
        expected_dpo_schema: str | None = None
        validated_rows = 0

        # Validate each line in the JSONL file. Log identifiers (path + row number)
        # only — never the raw line/object, which can contain customer training data.
        with open(file_path, "r", encoding="utf-8") as jsonl_file:
            for line_number, raw_line in enumerate(jsonl_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    obj: dict[str, Any] = json.loads(line)
                except Exception as e:
                    logger.debug(f"{file_path}:{line_number} is not valid JSON: {e}")
                    raise DatasetFormatError(f"{file_path} line {line_number} is not valid JSON: {e}")

                # Reject files that mix multiple DPO schemas: detect_dpo_schema_name()
                # later selects one concrete loader from the first row, so a mixed
                # file would silently be handed to the wrong NeMo-RL dataset class.
                if is_dpo:
                    row_schema = get_preference_dataset_discriminator(obj)
                    if expected_dpo_schema is None:
                        expected_dpo_schema = row_schema
                    elif row_schema != expected_dpo_schema:
                        raise DatasetFormatError(
                            f"{file_path} mixes DPO dataset schemas: expected {expected_dpo_schema}, "
                            f"got {row_schema} on line {line_number}"
                        )

                try:
                    self._validate_json_object(obj, validator)
                except Exception as e:
                    logger.debug(f"{file_path}:{line_number} does not conform to the expected schema: {e}")
                    raise DatasetFormatError(
                        f"{file_path} line {line_number} does not conform to the expected schema: {e}"
                    )
                validated_rows += 1
                if validated_rows % _VALIDATE_PROGRESS_INTERVAL == 0:
                    logger.info("Validated %s rows in %s", validated_rows, file_path)

        # A whitespace-only file would otherwise pass silently (no rows validated).
        if validated_rows == 0:
            raise DatasetFormatError(f"{file_path} has no non-empty rows to validate")
        logger.info("Validated %s rows in %s", validated_rows, file_path)


def _first_nonempty_line(file_path: str | Path) -> str | None:
    """Return the first non-blank line (stripped) of a file, or None if there is none."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def detect_dpo_schema_name(file_path: str | Path) -> str:
    """Detect the DPO preference dataset schema from the first line of the file.

    This function reads the first line of a JSONL dataset file and determines
    which preference dataset schema it matches. It's designed to be called after
    prepare_dataset() to dynamically determine the correct NeMo RL dataset class.

    For DPO training, it detects one of:
    - PreferenceDataset: Native format with context + ranked completions
    - BinaryPreferenceDataset: Simple prompt/chosen_response/rejected_response
    - HelpSteer3: NVIDIA HelpSteer3 format with preference scores
    - Tulu3Preference: AllenAI Tulu3 format with message lists

    Args:
        file_path: Path to the dataset file (JSONL format)

    Returns:
        The NeMo RL dataset class name (e.g., "BinaryPreferenceDataset", "HelpSteer3")

    Raises:
        DatasetFormatError: If the file is empty or not valid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise DatasetFormatError(f"Dataset file not found: {file_path}")

    # Read the first non-empty row so a leading blank line doesn't break detection.
    first_line = _first_nonempty_line(file_path)
    if first_line is None:
        raise DatasetFormatError(f"Dataset file has no content: {file_path}")

    # Parse as JSON
    try:
        obj: dict[str, Any] = json.loads(first_line)
    except json.JSONDecodeError as e:
        raise DatasetFormatError(f"First row of {file_path} is not valid JSON: {e}")

    # Use the discriminator function to detect the schema type (returns NeMo RL class name directly)
    dataset_name = get_preference_dataset_discriminator(obj)
    logger.debug(f"Detected DPO preference dataset: {dataset_name} for {file_path}")

    logger.info(f"Detected dataset schema '{dataset_name}' for {file_path}")
    return dataset_name


# Backward compatibility: provide standalone functions that create a validator instance
def detect_dataset_schema(
    file_path: str,
    training_type: TrainingType,
    *,
    finetuning_type: Optional[FinetuningType] = None,
    prompt_template: str | None = None,
) -> str:
    """Detect the dataset schema from the first line of the file."""
    validator = DatasetValidator(training_type, finetuning_type, prompt_template=prompt_template)
    return validator.detect_dataset_schema(file_path)


def validate_dataset(
    file_path: str,
    training_type: TrainingType,
    *,
    dataset_type: Optional[str] = None,
    finetuning_type: Optional[FinetuningType] = None,
    prompt_template: str | None = None,
) -> None:
    """Validate a single dataset file."""
    validator = DatasetValidator(training_type, finetuning_type, prompt_template=prompt_template)
    validator.validate_dataset(file_path, dataset_type)


def validate_datasets(
    file_names: list[str],
    training_type: TrainingType,
    *,
    dataset_type: Optional[str] = None,
    finetuning_type: Optional[FinetuningType] = None,
    prompt_template: str | None = None,
) -> None:
    """Validate a list of dataset files."""
    validator = DatasetValidator(training_type, finetuning_type, prompt_template=prompt_template)
    for file_name in file_names:
        validator.validate_dataset(file_name, dataset_type)
