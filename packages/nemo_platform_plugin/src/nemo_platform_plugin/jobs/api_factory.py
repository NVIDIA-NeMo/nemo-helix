# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import logging
import os
import tarfile
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Mapping
from datetime import datetime
from enum import StrEnum, auto
from functools import partial
from pathlib import Path
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

from anyio import open_file, to_thread
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_platform_plugin.api.parsed_filter import ParsedFilter, make_filter_dep
from nemo_platform_plugin.authz import AuthzScope, CallerKind, path_rule
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NemoHTTPError
from nemo_platform_plugin.client.types import RetryPolicy
from nemo_platform_plugin.dependencies import get_entity_client, get_sdk_client
from nemo_platform_plugin.entities import EntityClient
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.docker import validate_gpu_available_for_docker
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError, PlatformJobDependencyUnavailableError
from nemo_platform_plugin.jobs.openapi_utils import generate_openapi_extra_params
from nemo_platform_plugin.jobs.providers import (
    ComputeResources,
    ComputeResourceSpec,
    CPUExecutionProvider,
    DistributedGPUExecutionProvider,
    GPUExecutionProvider,
    Provider,
    SubprocessExecutionProvider,
)
from nemo_platform_plugin.jobs.providers import (
    ContainerSpec as ProviderContainerSpec,
)
from nemo_platform_plugin.jobs.result_manager import download_from_result_info
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobListResultResponse,
    PlatformJobLogPage,
    PlatformJobResultResponse,
    PlatformJobStatus,
    PlatformJobStatusResponse,
)
from nemo_platform_plugin.jobs.spec import (
    PlatformJobEnvironmentVariable,
    PlatformJobSecret,
    PlatformJobSecretEnvironmentVariableRef,
    PlatformJobSpec,
    PlatformJobStepSpec,
)
from nemo_platform_plugin.jobs.spec import (
    StepLifecycle as PlatformJobStepLifecycle,
)
from nemo_platform_plugin.jobs.telemetry import stamp_job_telemetry_plugins
from nemo_platform_plugin.jobs.types import (
    CreatePlatformJobRequest,
    JobLogsQueryParams,
    ListJobsQueryParams,
    validate_output_location,
)
from nemo_platform_plugin.jobs.types import (
    PlatformJobResponse as PlatformJob,
)
from nemo_platform_plugin.schema import DatetimeFilter, Filter, Page, PaginationData, StringFilter
from pydantic import BaseModel, Field, TypeAdapter, field_validator

logger = logging.getLogger(__name__)

# Public compatibility aliases for services that build Jobs specs through the
# route factory. These now point at plugin-owned Pydantic models, not Stainless
# generated SDK TypedDicts.
PlatformJobStep = PlatformJobStepSpec
ExecutorSpec = Provider
ContainerSpec = ProviderContainerSpec
CPUExecutionProviderSpec = CPUExecutionProvider
GPUExecutionProviderSpec = GPUExecutionProvider
DistributedGPUExecutionProviderSpec = DistributedGPUExecutionProvider
SubprocessExecutionProviderSpec = SubprocessExecutionProvider
ResourcesSpec = ComputeResources
ResourcesLimitsSpec = ComputeResourceSpec
ResourcesRequestsSpec = ComputeResourceSpec
StepLifecycle = PlatformJobStepLifecycle
EnvironmentVariable = PlatformJobEnvironmentVariable
EnvironmentVariableFromSecret = PlatformJobSecretEnvironmentVariableRef
JobSecret = PlatformJobSecret
PlatformJobSpecLike = PlatformJobSpec | Mapping[str, Any]

# Descriptions stamped onto the standard job permissions, keyed by verb. The catalog is
# derived from the routes, so these descriptions are the source of truth for the generated
# permission registry (no separate declaration to keep in sync).
_JOB_PERMISSION_DESCRIPTIONS: dict[str, str] = {
    "create": "Create {ns} jobs",
    "list": "List {ns} jobs",
    "read": "Read {ns} jobs, including status, logs, and results",
    "delete": "Delete {ns} jobs",
    "cancel": "Cancel {ns} jobs",
    "pause": "Pause {ns} jobs",
    "resume": "Resume {ns} jobs",
}

JobConfigT = TypeVar("JobConfigT", bound=BaseModel)
JobInputT = TypeVar("JobInputT", bound=BaseModel)
JobOutputT = TypeVar("JobOutputT", bound=BaseModel)

JobSchema = Type[BaseModel] | UnionType
JobSchemaAnnotated = Annotated[Any, object]
JobSchemaLike = JobSchema | JobSchemaAnnotated


class BaseJobRequest(BaseModel, Generic[JobConfigT]):
    name: str | None = None
    description: str | None = None
    project: str | None = None
    spec: JobConfigT
    profile: str | None = None
    options: dict[str, Any] | None = None
    ownership: dict | None = None
    custom_fields: dict | None = None
    output_location: str | None = None

    _validate_output_location = field_validator("output_location")(validate_output_location)


class BaseJob(BaseModel, Generic[JobConfigT]):
    id: str | None = None
    name: str
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: JobConfigT
    status: PlatformJobStatus | None = None
    status_details: dict[str, object] | None = None
    error_details: dict[str, object] | None = None
    ownership: dict[str, object] | None = None
    custom_fields: dict[str, object] | None = None


class BaseJobsListFilter(Filter):
    created_at: DatetimeFilter | None = Field(
        default=None, description="Jobs created at 'gte' datetime or 'lte' datetime."
    )
    name: StringFilter | str | None = Field(default=None, description="Name of the job.")
    workspace: str | None = Field(default=None, description="Workspace of the job.")
    project: str | None = Field(default=None, description="Project containing the job.")
    status: PlatformJobStatus | None = Field(default=None, description="The current status.")
    updated_at: DatetimeFilter | None = Field(
        default=None, description="Jobs updated at 'gte' datetime or 'lte' datetime."
    )


class BaseJobsSortField(StrEnum):
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"


def _make_jobs_sort_field(job_type: str) -> type[StrEnum]:
    return StrEnum(
        f"{job_type}JobsSortField",
        {name: member.value for name, member in BaseJobsSortField.__members__.items()},
    )


# Field-name allowlists for value-side validation in list_jobs. ``make_filter_dep``
# only validates field NAMES against ``BaseJobsListFilter.model_fields``; it does
# not enforce that ``status`` values are valid ``PlatformJobStatus`` literals or
# that ``created_at``/``updated_at`` operators are range comparisons. Without
# these checks, invalid status enum values silently filter to zero results in
# core jobs (dispatcher.py drops unknown literals) and bad datetime operators
# can 500 the entity store. Validate explicitly at the plugin boundary instead.
_DATETIME_FIELDS = frozenset({"created_at", "updated_at"})
_DATETIME_OPS = frozenset({FilterOperator.GTE, FilterOperator.LTE, FilterOperator.GT, FilterOperator.LT})
_STATUS_OPS = frozenset({FilterOperator.EQ, FilterOperator.IN, FilterOperator.NIN})


def _validate_jobs_filter_values(operation: FilterOperation) -> None:
    """Walk the operation tree and reject value/operator combos invalid for the jobs schema.

    - ``status``: each value must be a valid ``PlatformJobStatus`` literal; only
      equality / membership operators (``$eq``, ``$in``, ``$nin``) are allowed.
    - ``created_at`` / ``updated_at``: only range operators
      (``$gte``, ``$lte``, ``$gt``, ``$lt``) are allowed and the value must
      parse as ISO-8601.

    Raises ``ValueError`` on the first violation; the caller maps that to a
    400 ``HTTPException`` so the failure surfaces at the plugin boundary
    rather than as an empty result list or downstream 500.
    """
    if isinstance(operation, ComparisonOperation):
        field = operation.field
        if field == "status":
            if operation.operator not in _STATUS_OPS:
                raise ValueError(
                    f"Operator '{operation.operator.value}' is not supported on 'status'; "
                    f"use one of {sorted(op.value for op in _STATUS_OPS)}"
                )
            # Operator-specific shape: $in/$nin require a list; $eq requires a
            # scalar. Bracket notation comma-splits for $in/$nin via
            # _normalize_value, but raw JSON filters bypass that and can ship
            # mismatched shapes that pass the enum check by accident.
            is_membership = operation.operator in (FilterOperator.IN, FilterOperator.NIN)
            value_is_list = isinstance(operation.value, list)
            if is_membership and not value_is_list:
                raise ValueError(
                    f"Operator '{operation.operator.value}' on 'status' expects a list of values, "
                    f"got {type(operation.value).__name__}"
                )
            if not is_membership and value_is_list:
                raise ValueError(
                    f"Operator '{operation.operator.value}' on 'status' expects a scalar value, got a list"
                )
            values = operation.value if value_is_list else [operation.value]
            for v in values:
                if v is None:
                    continue
                try:
                    PlatformJobStatus(v)
                except ValueError as exc:
                    valid = sorted(s.value for s in PlatformJobStatus)
                    raise ValueError(f"Invalid status value '{v}'; valid values: {valid}") from exc
        elif field in _DATETIME_FIELDS:
            if operation.operator not in _DATETIME_OPS:
                raise ValueError(
                    f"Operator '{operation.operator.value}' is not supported on '{field}'; "
                    f"use one of {sorted(op.value for op in _DATETIME_OPS)}"
                )
            try:
                datetime.fromisoformat(str(operation.value).replace("Z", "+00:00"))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid datetime value '{operation.value}' for '{field}'") from exc
    elif isinstance(operation, LogicalOperation):
        for child in operation.operations:
            _validate_jobs_filter_values(child)


@overload
def handle_job_spec_mismatch(job_input: Type[JobConfigT], spec: object) -> JobConfigT: ...


@overload
def handle_job_spec_mismatch(job_input: JobSchemaLike, spec: object) -> BaseModel: ...


def handle_job_spec_mismatch(job_input: Type[JobConfigT] | JobSchemaLike, spec: object) -> JobConfigT | BaseModel:
    """
    Handle a job spec mismatch between the microservice job config spec and what is stored
    by the jobs microservice.
    """
    try:
        type_adapter = TypeAdapter(job_input)
        job_spec = type_adapter.validate_python(spec)
    except Exception as e:
        # What happens here is that we received a functional microservice job config back from
        # the jobs microservice that no longer can be mapped into the functional microservice's
        # job config model, making it not forward-compatible with the functional microservice.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to render job config: {str(e)}",
        )

    return job_spec


def _validate_job_spec(job: object) -> PlatformJobSpec:
    """Validate the job spec for common misconfigurations."""
    try:
        platform_job = PlatformJobSpec.model_validate(job)
    except Exception as e:
        raise PlatformJobCompilationError(f"platform job spec is invalid: {str(e)}") from e

    # Validate that any config's provided are serializeable to json
    try:
        for step in platform_job.steps:
            json.dumps(step.config)
    except Exception as e:
        raise PlatformJobCompilationError(f"step config is not json serializable: {str(e)}")

    validate_gpu_available_for_docker(platform_job)
    return platform_job


class BaseResultSerializer(BaseModel, ABC):
    def route_kwargs(self) -> dict:
        """
        These kwargs are passed into `router.add_api_route`. Modifying these values
        can be useful to coerce fastapi/openapi about the proper type of the
        data that is being serialized.
        """
        return {
            "responses": {
                status.HTTP_200_OK: {"description": "Successful Response"},
                status.HTTP_404_NOT_FOUND: {"description": "Not Found"},
            }
        }

    @abstractmethod
    def serialize(self, output_path: Path, limit: int | None = None) -> Response:
        """
        Convert a file/dir into an appropriate fastapi output format.
        """
        raise NotImplementedError()


class JSONResultSerializer(BaseResultSerializer):
    serializer_type: Literal["json"] = "json"

    def serialize(self, output_path: Path, limit: int | None = None) -> JSONResponse:
        del limit
        with output_path.open() as f:
            return JSONResponse(json.load(f))


class JSONLResultSerializer(BaseResultSerializer):
    serializer_type: Literal["jsonl"] = "jsonl"

    def route_kwargs(self) -> dict:
        """
        Configure API endpoint with streaming response and support `limit` query parameter
        """
        ret: dict = super().route_kwargs() | {"response_class": StreamingResponse}
        # Explicitly describe that the 200s return JSONL.
        ret["responses"][status.HTTP_200_OK]["content"] = {
            "application/jsonl": {"schema": {"type": "object", "additionalProperties": True}},
        }
        return ret

    async def _file_lines_iterator(self, file_path: Path, limit: int | None = None):
        async with await open_file(file_path, mode="r", encoding="utf-8") as f:
            lines = 0
            async for line in f:
                yield line
                lines += 1
                if limit and lines >= limit:
                    return

    def serialize(self, output_path: Path, limit: int | None = None, **_kwargs: object) -> StreamingResponse:
        """
        Stream a DataFrame row by row as JSON objects
        """
        return StreamingResponse(self._file_lines_iterator(output_path, limit), media_type="application/json")


class PydanticJSONLResultSerializer(BaseResultSerializer):
    serializer_type: Literal["pydantic_jsonl"] = "pydantic_jsonl"
    model: Type[BaseModel]
    serialize_kwargs: dict = Field(default_factory=dict)

    def route_kwargs(self) -> dict:
        # Set response_model so FastAPI includes the model schema in OpenAPI components.
        ret: dict = super().route_kwargs() | {"response_class": StreamingResponse, "response_model": self.model}
        ret["responses"][status.HTTP_200_OK]["content"] = {
            "application/jsonl": {"schema": {"$ref": f"#/components/schemas/{self.model.__name__}"}},
        }
        return ret

    async def _validated_lines_iterator(self, file_path: Path, limit: int | None = None):
        lines = 0
        line_number = 0
        async with await open_file(file_path, mode="r", encoding="utf-8") as f:
            async for line in f:
                line_number += 1
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    json_in = json.loads(stripped)
                    inst = self.model.model_validate(json_in)
                    json_out = inst.model_dump(mode="json", **self.serialize_kwargs)
                except Exception as e:
                    logger.exception("Failed to validate JSONL stream line", extra={"line_number": line_number})
                    yield (
                        json.dumps(
                            {
                                "error": {
                                    "type": e.__class__.__name__,
                                    "line": line_number,
                                    "message": "Failed to validate result line; see server logs for details.",
                                }
                            }
                        )
                        + "\n"
                    )
                    return

                yield json.dumps(json_out) + "\n"
                lines += 1
                if limit and lines >= limit:
                    return

    def serialize(self, output_path: Path, limit: int | None = None) -> StreamingResponse:
        return StreamingResponse(self._validated_lines_iterator(output_path, limit), media_type="application/jsonl")


class PydanticResultSerializer(BaseResultSerializer):
    serializer_type: Literal["pydantic"] = "pydantic"
    model: Type[BaseModel]
    serialize_kwargs: dict = Field(default_factory=dict)

    def route_kwargs(self) -> dict:
        # Setting `response_model` tells fastapi the proper object
        # to use when generating the schema.
        return super().route_kwargs() | {"response_model": self.model}

    def serialize(self, output_path: Path, limit: int | None = None) -> JSONResponse:
        del limit
        with output_path.open() as f:
            json_in = json.load(f)

        try:
            inst = self.model.model_validate(json_in)
            json_out = inst.model_dump(mode="json", **self.serialize_kwargs)
        except ValueError:
            logger.warning("Failed to load/dump json; returning result's raw json", exc_info=True)
            json_out = json_in

        return JSONResponse(json_out)


class FileResultSerializer(BaseResultSerializer):
    serializer_type: Literal["file"] = "file"

    def route_kwargs(self) -> dict:
        # Without `response_class`, fastapi will think this route returns json too.
        ret: dict = super().route_kwargs() | {"response_class": FileResponse}

        # Explicitly describe that the 200s return octet-stream binary data.
        ret["responses"][status.HTTP_200_OK]["content"] = {
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
        }
        return ret

    def serialize(self, output_path: Path, limit: int | None = None) -> FileResponse:
        del limit
        if output_path.is_dir():
            filename = f"{output_path.name}.tar.gz"
            tar_path = output_path.parent / filename
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(output_path, arcname=os.path.basename(output_path))
            return FileResponse(path=output_path, filename=filename)
        else:
            return FileResponse(path=output_path, filename=output_path.name)


ResultSerializer = Annotated[
    JSONResultSerializer
    | JSONLResultSerializer
    | PydanticJSONLResultSerializer
    | FileResultSerializer
    | PydanticResultSerializer,
    Field(discriminator="serializer_type"),
]


class JobRouteOption(StrEnum):
    """Options for which job routes to enable in the generated job router."""

    CORE = auto()
    PAUSE_RESUME = auto()


class PlatformJobResultRoute(BaseModel):
    name: str
    serializer: ResultSerializer


# Compiler types: compiler receives both input spec (user-provided) and output spec (with auto-generated fields)
# Signature: (workspace, original_spec, transformed_spec, entity_client, job_name, sdk) -> PlatformJobSpec
# job_name is the resolved name (user-provided or auto-generated), None when no name is available
# sdk is always provided for accessing secrets, files, and models with user context
# A compiler may be implemented synchronously or asynchronously. Model that in
# the callable's return type instead of narrowing the callable itself with
# runtime-only typing helpers.
PlatformJobSpecCompiler = Callable[
    [str, JobInputT, JobOutputT, EntityClient, str | None, AsyncNeMoPlatform],
    PlatformJobSpecLike | Awaitable[PlatformJobSpecLike],
]
PlatformJobSpecCompilerAsync = Callable[
    [str, JobInputT, JobOutputT, EntityClient, str | None, AsyncNeMoPlatform], Awaitable[PlatformJobSpecLike]
]


# Input-to-output transformer types: receives job_name to use for related fields (e.g., output)
# Signature: (original_spec, workspace, entity_client, job_name, sdk) -> transformed_spec
InputToOutputTransformer = Callable[[JobInputT, str, EntityClient, str | None, AsyncNeMoPlatform], JobOutputT]
InputToOutputTransformerAsync = Callable[
    [JobInputT, str, EntityClient, str | None, AsyncNeMoPlatform], Awaitable[JobOutputT]
]

# Job name generator: called when user doesn't provide a name
JobNameGenerator = Callable[[], str]


def _unwrap_annotated_schema(obj: object) -> object:
    if get_origin(obj) is Annotated:
        return get_args(obj)[0]
    return obj


def _accepts_entity_client(func: Callable) -> bool:
    """Check if function signature includes entity_client parameter."""
    sig = inspect.signature(func)
    return "entity_client" in sig.parameters or "entities_client" in sig.parameters


def _is_union_type(obj: object) -> bool:
    origin = get_origin(obj)
    return origin is Union or isinstance(obj, UnionType)


def _is_basemodel_union(obj: object) -> bool:
    """
    Runtime validation that checks if obj is a UnionType where all members
    are BaseModel subclasses.
    """
    if not _is_union_type(obj):
        return False
    for union_arg in get_args(obj):
        candidate = _unwrap_annotated_schema(union_arg)
        if not (isinstance(candidate, type) and issubclass(candidate, BaseModel)):
            return False
    return True


def _validate_basemodel_or_union(
    obj: JobSchemaLike,
    param_name: str,
) -> None:
    """Validate that obj is a BaseModel or a Union of BaseModel subclasses.

    Args:
        obj: The type to validate
        param_name: Name of the parameter (used in error messages)

    Raises:
        ValueError: If obj is not a valid BaseModel or Union of BaseModel subclasses
    """
    base = _unwrap_annotated_schema(obj)
    if _is_union_type(base):
        if not _is_basemodel_union(base):
            raise ValueError(f"{param_name} must be a BaseModel or a Union of BaseModel subclasses, got {obj!r}")
        return

    if not (isinstance(base, type) and issubclass(base, BaseModel)):
        raise ValueError(f"{param_name} must be a BaseModel or a Union of BaseModel subclasses, got {obj!r}")


def _validate_and_resolve_job_output(
    job_output: JobSchemaLike | None,
    job_input: JobSchemaLike,
    input_to_output: InputToOutputTransformer | InputToOutputTransformerAsync | None,
) -> tuple[JobSchemaLike, InputToOutputTransformer | InputToOutputTransformerAsync | None]:
    """Validate and resolve job_output parameter.

    Handles defaulting and validation of the job_output parameter, which determines
    the schema used for storing and returning job data.

    Args:
        job_output: The job output schema (what gets stored and returned).
            If None, defaults to job_input for backward compatibility.
        job_input: The job input schema (used as default when job_output is None)
        input_to_output: Transformer function that converts input to output schema.
            Required when job_output is provided.

    Returns:
        Tuple of (resolved_job_output, input_to_output) where:
        - resolved_job_output: The actual output schema to use (either job_output or job_input)
        - input_to_output: The input to output transformer to use (either input_to_output or None)

    Raises:
        ValueError: If job_output is provided but input_to_output is None
        ValueError: If job_output is not a valid BaseModel or Union of BaseModel subclasses
        ValueError: If input_to_output is provided when job_output is not provided
    """
    if job_output is None and input_to_output is not None:
        raise ValueError("input_to_output parameter must not be provided when job_output is not provided.")

    if job_output is None:
        # Default: use job_input for both input and output (backward compatible)
        return job_input, None

    # Validate job_output type first so callers see type errors before missing-transformer errors
    _validate_basemodel_or_union(job_output, "job_output")

    # Validate input_to_output is provided
    if input_to_output is None:
        raise ValueError(
            "input_to_output parameter is required when job_output is provided. "
            "This transformer converts the input schema to the output schema."
        )

    return job_output, input_to_output


def _resolve_job_name(
    user_provided_name: str | None,
    generate_job_name: JobNameGenerator | None,
) -> str | None:
    """Resolve the job name from a user-provided name or a generator callback.

    Priority:
        1. User-provided name (request_name)
        2. Auto-generated name via generate_job_name callback
        3. None (Jobs Service will auto-generate)

    This name is passed to `input_to_output` and `platform_job_config_compiler` so that it can be used
    for relevant fields as needed.
    """
    if user_provided_name is not None:
        return user_provided_name
    if generate_job_name is not None:
        return generate_job_name()
    return None


async def _transform_input_to_output(
    input_to_output: InputToOutputTransformer | InputToOutputTransformerAsync | None,
    spec: JobSchemaLike,
    workspace: str,
    entity_client: EntityClient,
    job_name: str | None,
    service_name: str,
    sdk: AsyncNeMoPlatform,
) -> JobSchemaLike:
    """Transform a job input spec into an output spec using the provided transformer.

    If no transformer is provided, returns the input spec unchanged (backward compatible).
    Supports both sync and async transformer callables.

    Raises:
        HTTPException(422): If the transformer raises any other exception.
        PermissionError: If the transformer raises a PermissionError.
    """
    if input_to_output is None:
        return spec
    try:
        if inspect.iscoroutinefunction(input_to_output):
            return await input_to_output(spec, workspace, entity_client, job_name, sdk)
        # Run sync transformers in a thread pool to avoid blocking the event loop.
        return await to_thread.run_sync(partial(input_to_output, spec, workspace, entity_client, job_name, sdk))
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Failed to transform {service_name} job input to output: {str(e)}",
        ) from e


async def _compile_platform_spec(
    compiler: PlatformJobSpecCompiler,
    workspace: str,
    original_spec: JobSchemaLike,
    transformed_spec: JobSchemaLike,
    entity_client: EntityClient,
    job_name: str | None,
    service_name: str,
    sdk: AsyncNeMoPlatform,
    profile: str | None,
    options: dict[str, Any] | None,
) -> PlatformJobSpec:
    """Compile input and output specs into a PlatformJobSpec for execution.

    The compiler receives both the input spec (user-provided fields) and output spec
    (with auto-generated fields), allowing it to distinguish between user intent and
    system-generated values.

    Supports both sync and async compiler callables. Validates the resulting
    spec for common misconfigurations.

    Raises:
        HTTPException(422): If the compiler raises PlatformJobCompilationError.
        HTTPException(503): If the compiler raises PlatformJobDependencyUnavailableError.
        PermissionError: If the compiler raises a PermissionError.
    """
    try:
        submit_control_kwargs = _submit_control_kwargs(compiler, profile=profile, options=options)
        compile_result: PlatformJobSpecLike | Awaitable[PlatformJobSpecLike]
        if inspect.iscoroutinefunction(compiler):
            compile_result = compiler(
                workspace, original_spec, transformed_spec, entity_client, job_name, sdk, **submit_control_kwargs
            )
        else:
            # Run sync compilers in a thread pool to avoid blocking the event loop.
            compile_result = await to_thread.run_sync(
                partial(
                    compiler,
                    workspace,
                    original_spec,
                    transformed_spec,
                    entity_client,
                    job_name,
                    sdk,
                    **submit_control_kwargs,
                )
            )
        if inspect.isawaitable(compile_result):
            platform_spec = await compile_result
        else:
            platform_spec = compile_result

        return _validate_job_spec(platform_spec)
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e
    except PlatformJobDependencyUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Temporarily unable to compile {service_name} job spec: {str(e)}",
        ) from e
    except PlatformJobCompilationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Failed to compile {service_name} job spec: {str(e)}",
        ) from e


def _submit_control_kwargs(
    compiler: Callable[..., object],
    *,
    profile: str | None,
    options: dict[str, Any] | None,
) -> dict[str, object]:
    """Return profile/options kwargs accepted by *compiler*.

    The low-level ``job_route_factory`` predates submitter controls and has
    six-argument compiler call sites. Preserve those call sites while allowing
    ``add_job_routes`` adapters and newer compilers to receive the body fields.
    """
    signature = inspect.signature(compiler)
    parameters = signature.parameters
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    accepts_profile = accepts_kwargs or "profile" in parameters
    accepts_options = accepts_kwargs or "options" in parameters

    unsupported = []
    if profile is not None and not accepts_profile:
        unsupported.append("profile")
    if options and not accepts_options:
        unsupported.append("options")
    if unsupported:
        compiler_name = getattr(compiler, "__qualname__", getattr(compiler, "__name__", repr(compiler)))
        raise PlatformJobCompilationError(f"{compiler_name} does not support submit field(s): {', '.join(unsupported)}")

    kwargs: dict[str, object] = {}
    if accepts_profile:
        kwargs["profile"] = profile
    if accepts_options:
        kwargs["options"] = options
    return kwargs


def job_route_factory(
    service_name: str,
    job_type: str,
    job_input: JobSchemaLike,
    platform_job_config_compiler: PlatformJobSpecCompiler[JobInputT, JobOutputT],
    route_options: list[JobRouteOption] | None = None,
    job_result_routes: list[PlatformJobResultRoute] | None = None,
    job_output: JobSchemaLike | None = None,
    input_to_output: InputToOutputTransformer | InputToOutputTransformerAsync | None = None,
    generate_job_name: JobNameGenerator | None = None,
    authz: AuthzScope | None = None,
) -> APIRouter:
    """Create a job router with standard CRUD operations.

    The SDK is injected per-request via FastAPI dependency injection, which ensures
    that user auth headers are properly propagated to the jobs service.

    Args:
        service_name: Name of the microservice (e.g., "customization").
        job_type: Type prefix for generated schema names (e.g., "Customization").
        job_input: The job input schema (what users provide in POST requests body.spec field).
        platform_job_config_compiler: Compiles job specs to PlatformJobSpec for execution.
            Signature: (workspace, original_spec, transformed_spec, entity_client, job_name, sdk) -> PlatformJobSpec.
            Receives transformed_spec (which contains all input fields plus auto-generated fields)
            and the resolved job_name for configuring outputs.
            job_name is None when no name was provided and no generator exists.
        route_options: Which routes to enable (defaults to CORE).
        job_result_routes: Custom result download routes.
        job_output: The job output schema (what gets stored and returned).
            If not provided, defaults to job_input (same type for input/output).
        input_to_output: Transforms job input to job output.
            Signature: (original_spec, workspace, entity_client, job_name, sdk) -> transformed_spec.
            Called on create to add auto-generated fields. Receives job_name so it can
            use the same name for related fields (e.g., output).
        generate_job_name: Called when user doesn't provide a job name. Returns the
            auto-generated name to use.

    Example with separate input/output types:
        ```python
        def input_to_output(
            original_spec: CustomizationJobInput,
            workspace: str,
            entity_client: EntityClient,
            job_name: str,
            sdk: AsyncNeMoPlatform,
        ) -> CustomizationJobOutput:
            return CustomizationJobOutput(
                ...
            )

        def platform_job_config_compiler(
            workspace: str,
            original_spec: CustomizationJobInput,
            transformed_spec: CustomizationJobOutput,
            entity_client: EntityClient,
            job_name: str,
            sdk: AsyncNeMoPlatform,
        ) -> PlatformJobSpec:
            ...

        router = job_route_factory(
            service_name="customization",
            job_type="Customization",
            job_input=CustomizationJobInput,
            job_output=CustomizationJobOutput,
            input_to_output=transform_input,
            platform_job_config_compiler=compile_platform_spec,
            generate_job_name=generate_customization_id,
        )
        ```
    """
    _validate_basemodel_or_union(job_input, "job_input")

    # Handle job_output defaulting and validation
    job_output, input_to_output = _validate_and_resolve_job_output(job_output, job_input, input_to_output)

    if route_options is None:
        route_options = [JobRouteOption.CORE]

    router = APIRouter()
    service_name = service_name.lower()

    def _stamp(endpoint: Callable[..., Any], *, perm: str, write: bool) -> Callable[..., Any]:
        """Attach a PRINCIPAL ``@path_rule`` to a generated job route.

        Inert unless the caller passed an ``authz`` scope — so unmigrated callers keep
        emitting unauthz'd routes (handled by the bundle fail-mode). Returns *endpoint*
        so it can wrap download closures inline.
        """
        if authz is not None:
            permission = authz.permission(
                perm,
                description=_JOB_PERMISSION_DESCRIPTIONS[perm].format(ns=authz.namespace),
            )
            path_rule(callers=[CallerKind.PRINCIPAL], permissions=[permission])(endpoint)
            # Scope is declared separately from the permission rule (see authz.AuthzScope).
            (authz.write if write else authz.read)(endpoint)
        return endpoint

    # These lines dynamically create new classes, named for the client microservice
    # using the job route factory, for use as input and output types in the FastAPI routes.
    # This style (using `type` with three args) ensures the class is named properly, i.e.
    # with `job_type` as a prefix, so that the correct name is used in the openapi spec, instead of
    # multiple duplicative `TypedJob` and `TypedJobRequest` definitions (one for each factory client).
    # The first arg is the type/class name, the second arg is a tuple of base classes to inherit,
    # and the third arg defines the runtime ``spec`` annotation used by Pydantic.
    #
    # Request uses job_input (input schema), Response uses job_output (output schema)
    # When job_output is not provided, they are the same type.
    # _validate_basemodel_or_union already validated these types are either
    # BaseModel subclasses or unions of BaseModel subclasses.
    TypedJobRequest = type(
        f"{job_type}JobRequest",
        (BaseJobRequest[BaseModel],),
        {"__annotations__": {"spec": job_input}},
    )
    TypedJobResponse = type(
        f"{job_type}Job",
        (BaseJob[BaseModel],),
        {"__annotations__": {"spec": job_output}},
    )
    TypedJobsListFilter = type(f"{job_type}JobsListFilter", (BaseJobsListFilter,), {})
    # Keep the OpenAPI schema names stable for generated SDK consumers.  The
    # values are shared across job factories, but Studio imports the generated
    # per-job symbols (for example ``EvaluateJobsSortField``), so the route
    # annotation must expose a concrete enum named for this job type.
    TypedJobsSortField = _make_jobs_sort_field(job_type)

    def from_response(job_resp: PlatformJob) -> TypedJobResponse:
        # Use job_output for deserialization (what's stored and returned)
        return TypedJobResponse(
            id=job_resp.id,
            name=job_resp.name,
            description=job_resp.description,
            project=None,
            workspace=job_resp.workspace,
            created_at=job_resp.created_at,
            updated_at=job_resp.updated_at,
            spec=handle_job_spec_mismatch(job_output, job_resp.spec),
            status=job_resp.status,
            status_details=job_resp.status_details,
            error_details=job_resp.error_details,
            ownership=job_resp.ownership,
            custom_fields=job_resp.custom_fields,
        )

    # Core job routes
    if JobRouteOption.CORE in route_options:

        @router.post(
            "/jobs",
            status_code=status.HTTP_201_CREATED,
        )
        async def create_job(
            workspace: str,
            request: TypedJobRequest,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
            entity_client: EntityClient = Depends(get_entity_client),
        ) -> TypedJobResponse:
            f"""Create a new job for the {service_name} microservice."""

            job_name = _resolve_job_name(request.name, generate_job_name)
            job_spec = await _transform_input_to_output(
                input_to_output,
                request.spec,
                workspace,
                entity_client,
                job_name,
                service_name,
                sdk,
            )
            platform_spec = await _compile_platform_spec(
                platform_job_config_compiler,
                workspace,
                request.spec,
                job_spec,
                entity_client,
                job_name,
                service_name,
                sdk,
                request.profile,
                request.options,
            )

            # Create the job using the SDK pointed to the platform jobs microservice.
            # Build SDK call kwargs, only including optional fields when they have values
            # (passing None explicitly causes different serialization than omitting)
            # Note: We store transformed_spec (not input), which includes auto-generated fields.
            # ``job_spec`` may be a Pydantic model (the transformed job output);
            # the request body's ``spec`` is a plain dict on the wire. The
            # Stainless SDK serialized models implicitly — the typed client
            # validates the body first, so coerce to a dict here.
            #
            # Dump ``mode="json"`` (not python): the spec is bound for JSON transport and JSON storage,
            # and the request body's ``spec`` is an opaque ``dict`` that has lost the nested models'
            # serializers. A python-mode dump leaves non-JSON-native values (e.g. a cloudpickle metric
            # bundle's raw ``bytes``, whose ``ser_json_bytes="base64"`` never runs) that then fail to
            # JSON-encode on the wire (invalid utf-8 on the pickle marker). ``mode="json"`` runs each
            # nested model's own JSON serializer, so bytes base64-encode and round-trip on read.
            spec_dict = job_spec.model_dump(mode="json") if isinstance(job_spec, BaseModel) else job_spec

            # Only include optional fields when they have values — passing None
            # explicitly serializes differently than omitting (exclude_unset).
            create_fields: dict = {
                "source": service_name,
                "spec": spec_dict,
                "platform_spec": platform_spec,
            }
            # Use the resolved job_name (user-provided or generated)
            if job_name is not None:
                create_fields["name"] = job_name
            if request.description is not None:
                create_fields["description"] = request.description
            if request.ownership is not None:
                create_fields["ownership"] = request.ownership
            custom_fields = stamp_job_telemetry_plugins(request.custom_fields, service_name)
            if custom_fields:
                create_fields["custom_fields"] = custom_fields
            if request.project:
                create_fields["project"] = request.project
            if request.output_location is not None:
                create_fields["output_location"] = request.output_location

            jobs = client_from_platform(sdk, AsyncJobsClient)
            job_resp = (
                await jobs.create_job(workspace=workspace, body=CreatePlatformJobRequest(**create_fields))
            ).data()
            return from_response(job_resp)

        async def list_jobs(
            workspace: str,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
            page: int = Query(default=1, description="Page number.", gt=0),
            page_size: int = Query(default=10, description="Page size.", gt=0),
            sort: BaseJobsSortField = Query(
                default=BaseJobsSortField.CREATED_AT_DESC,
                description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
            ),
            parsed: ParsedFilter = Depends(make_filter_dep(TypedJobsListFilter)),
        ) -> Page[TypedJobResponse]:
            f"""List all jobs for the {service_name} microservice."""

            # Enforce schema-level value validation (status enum, datetime
            # operators) on the parsed tree. ``make_filter_dep`` only checks
            # field names against ``BaseJobsListFilter.model_fields``; it does
            # not validate values, so without this check invalid status enums
            # silently filter to nothing in core jobs and bad datetime ops can
            # 500 the entity store.
            if parsed.operation is not None:
                try:
                    _validate_jobs_filter_values(parsed.operation)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=str(exc),
                    ) from exc
            # Capture the user filter for the API response BEFORE we add the
            # service-source predicate, so the response echoes the user's query.
            user_filter = parsed.to_response()
            # Compose an explicit AND of the user filter and the source predicate.
            # A flat dict merge {**user, "source": ...} silently drops "source"
            # when the user filter has a logical root ($or/$and/$not), since the
            # downstream parser short-circuits on the first logical operator.
            parsed.and_with(ComparisonOperation(operator=FilterOperator.EQ, field="source", value=service_name))
            # Serialize as JSON and forward via extra_query. The SDK's typed
            # ``filter`` param flows through a deep-object querystring serializer
            # whose ``comma`` array_format mangles list-of-dict values that
            # logical operators ($and/$or/$not) produce — joining them as
            # comma-separated Python reprs. core jobs' make_filter_dep already
            # accepts raw JSON in ``filter=`` and routes it through
            # parse_json_filter, so a single JSON-string param survives the
            # round trip cleanly.
            list_query: ListJobsQueryParams = {
                "page": page,
                "page_size": page_size,
                "sort": str(sort),
                "filter": json.dumps(parsed.to_response()),
            }
            jobs = client_from_platform(sdk, AsyncJobsClient)
            list_page = (await jobs.list_jobs(workspace=workspace, query_params=list_query)).page()
            return Page(
                data=[from_response(job) for job in list_page.items],
                pagination=PaginationData.model_validate(list_page.metadata),
                sort=sort,
                filter=user_filter or None,
            )

        list_jobs.__annotations__["sort"] = TypedJobsSortField
        router.add_api_route(
            "/jobs",
            list_jobs,
            methods=["GET"],
            response_model=Page[TypedJobResponse],
            response_model_exclude_none=True,
            openapi_extra=generate_openapi_extra_params(
                filter_schema=TypedJobsListFilter,
                filter_description="Filter jobs on various criteria.",
            ),
        )

        @router.get(
            "/jobs/{name}",
        )
        async def get_job(
            workspace: str,
            name: str,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> TypedJobResponse:
            f"""Get a job by name for the {service_name} microservice."""

            job_resp = (await client_from_platform(sdk, AsyncJobsClient).get_job(name=name, workspace=workspace)).data()
            return from_response(job_resp)

        # Status
        @router.get(
            "/jobs/{name}/status",
        )
        async def get_job_status(
            workspace: str,
            name: str,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> PlatformJobStatusResponse:
            f"""Get the status of a job by name for the {service_name} microservice."""
            job_resp = (
                await client_from_platform(sdk, AsyncJobsClient).get_job_status(name=name, workspace=workspace)
            ).data()
            return PlatformJobStatusResponse(**job_resp.model_dump())

        @router.delete(
            "/jobs/{name}",
            status_code=status.HTTP_204_NO_CONTENT,
            responses={
                status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
                status.HTTP_409_CONFLICT: {"description": "Job is not in a terminal state"},
            },
        )
        async def delete_job(
            workspace: str,
            name: str,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> None:
            f"""Delete a job by name for the {service_name} microservice."""
            try:
                jobs_client = client_from_platform(sdk, AsyncJobsClient).with_retry(RetryPolicy(max_retries=0))
                await jobs_client.delete_job(name=name, workspace=workspace)
            except NemoHTTPError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
            return None

        @router.post(
            "/jobs/{name}/cancel",
        )
        async def cancel_job(
            workspace: str,
            name: str,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> TypedJobResponse:
            f"""Cancel a job by name for the {service_name} microservice."""

            job_resp = (
                await client_from_platform(sdk, AsyncJobsClient).cancel_job(name=name, workspace=workspace)
            ).data()
            return from_response(job_resp)

        # Logs
        @router.get(
            "/jobs/{name}/logs",
        )
        async def get_job_logs(
            workspace: str,
            name: str,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
            limit: int | None = Query(default=None),
            page_cursor: str | None = Query(default=None),
            tail: int | None = Query(default=None, gt=0, le=10_000),
        ) -> PlatformJobLogPage:
            f"""Get the logs of a job by name for the {service_name} microservice."""

            logs_query: JobLogsQueryParams = {}
            if limit is not None:
                logs_query["limit"] = limit
            if page_cursor is not None:
                logs_query["page_cursor"] = page_cursor
            if tail is not None:
                logs_query["tail"] = tail
            logs_page = (
                await client_from_platform(sdk, AsyncJobsClient).list_job_logs(
                    workspace=workspace, name=name, query_params=logs_query
                )
            ).page()
            return PlatformJobLogPage(data=logs_page.items, **logs_page.metadata)

        # Results
        @router.get(
            "/jobs/{name}/results",
        )
        async def list_job_results(
            workspace: str,
            name: str,
            request: Request,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> PlatformJobListResultResponse:
            f"""Get the results of a job by name for the {service_name} microservice."""

            results = (
                await client_from_platform(sdk, AsyncJobsClient).list_job_results(name=name, workspace=workspace)
            ).data()
            result_dicts = [result.model_dump() for result in results.data]
            list_results = []
            for result_dict in result_dicts:
                result_name = result_dict["name"]
                result_dict["download_url"] = f"{request.url}/{result_name}/download"
                list_results.append(PlatformJobResultResponse(**result_dict))

            return PlatformJobListResultResponse(data=list_results)

        @router.get(
            "/jobs/{job}/results/{name}",
        )
        async def get_job_result(
            workspace: str,
            job: str,
            name: str,
            request: Request,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> PlatformJobResultResponse:
            f"""Get the result of a job by name for the {service_name} microservice."""

            result_obj = (
                await client_from_platform(sdk, AsyncJobsClient).get_job_result(name=name, job=job, workspace=workspace)
            ).data()

            # Construct the URL for downloading this result
            result_dict = result_obj.model_dump()
            result_dict["download_url"] = f"{request.url}/download"
            return PlatformJobResultResponse(**result_dict)

        # Stamp authorization rules on the generated routes (PRINCIPAL caller). Reads use
        # one shared <ns>.read permission; mutating routes get their own permission.
        _stamp(create_job, perm="create", write=True)
        _stamp(list_jobs, perm="list", write=False)
        _stamp(get_job, perm="read", write=False)
        _stamp(get_job_status, perm="read", write=False)
        _stamp(delete_job, perm="delete", write=True)
        _stamp(cancel_job, perm="cancel", write=True)
        _stamp(get_job_logs, perm="read", write=False)
        _stamp(list_job_results, perm="read", write=False)
        _stamp(get_job_result, perm="read", write=False)

        # Result downloads:
        # Services that use the api factory can utilize `job_result_routes` to map specific
        # `result_names`s to differently shaped objects. This can make the generated SDK smarter
        # about how to interpret the artifact that was created by the job. By default, the artifact
        # will just come back as a file or tar (if the data is a directory).
        if job_result_routes is None:
            job_result_routes = []

        async def _download_route_helper(
            workspace: str,
            name: str,
            job: str,
            background_tasks: BackgroundTasks,
            result_serializer: ResultSerializer,
            sdk: AsyncNeMoPlatform,
            **kwargs,
        ) -> Response:
            """
            This is the primary business-logic function that actually handles downloading the data
            from Jobs + datastore.
            - Fetch the result from the jobs store to make sure it exists
            - Use the `artifact_url` to download directly from the configured datastore
            - Use the `result_serializer` to know how to properly serialize the output
            """

            result_info = (
                await client_from_platform(sdk, AsyncJobsClient).get_job_result(name=name, job=job, workspace=workspace)
            ).data()
            _, tmp_dir_path = await download_from_result_info(
                result_name=name,
                job_name=job,
                workspace=workspace,
                artifact_url=result_info.artifact_url,
                files_client=client_from_platform(sdk, AsyncFilesClient),
            )
            background_tasks.add_task(lambda: tmp_dir_path.cleanup_tmp_dir())
            return result_serializer.serialize(tmp_dir_path.path)

        def _make_explicit_download_endpoint(job_result_route: PlatformJobResultRoute):
            """
            The below route function needs to take in only `job`, because fastapi/openapi
            use those arguments to determine the schema for this request.
            `result` will still be accessible due to the closure this function creates.
            """

            if isinstance(job_result_route.serializer, JSONLResultSerializer | PydanticJSONLResultSerializer):
                # Initialize route signature for serializer with query parameter
                async def jsonl_route(
                    workspace: str,
                    job: str,
                    background_tasks: BackgroundTasks,
                    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
                    limit: int | None = None,
                ) -> Response:
                    return await _download_route_helper(
                        workspace=workspace,
                        name=job_result_route.name,
                        job=job,
                        background_tasks=background_tasks,
                        result_serializer=job_result_route.serializer,
                        sdk=sdk,
                        limit=limit,
                    )

                return jsonl_route

            async def route(
                workspace: str,
                job: str,
                background_tasks: BackgroundTasks,
                sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
            ) -> Response:
                return await _download_route_helper(
                    workspace=workspace,
                    name=job_result_route.name,
                    job=job,
                    background_tasks=background_tasks,
                    result_serializer=job_result_route.serializer,
                    sdk=sdk,
                )

            return route

        def _make_generic_download_endpoint(result_serializer: ResultSerializer):
            """
            This route function is the generic catch-all endpoint for result routes that weren't
            specifically declared. For this function, we *do* want `result` to be a variable
            exposed in the path.
            """

            async def route(
                workspace: str,
                job: str,
                name: str,
                background_tasks: BackgroundTasks,
                sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
            ) -> Response:
                return await _download_route_helper(
                    workspace=workspace,
                    name=name,
                    job=job,
                    background_tasks=background_tasks,
                    result_serializer=result_serializer,
                    sdk=sdk,
                )

            return route

        for job_result_route in job_result_routes:
            router.add_api_route(
                name=f"download_job_result_{job_result_route.name}",
                path=f"/jobs/{{job}}/results/{job_result_route.name}/download",
                endpoint=_stamp(_make_explicit_download_endpoint(job_result_route), perm="read", write=False),
                **job_result_route.serializer.route_kwargs(),
            )

        # Add one final route for wildcard `{name}`, for undeclared results.
        # This route will simply return the result's artifact as a file.
        file_result_serializer = FileResultSerializer()
        router.add_api_route(
            name="download_job_result",
            path="/jobs/{job}/results/{name}/download",
            endpoint=_stamp(_make_generic_download_endpoint(file_result_serializer), perm="read", write=False),
            **file_result_serializer.route_kwargs(),
        )

    # Some jobs can be paused and resumed.
    if JobRouteOption.PAUSE_RESUME in route_options:

        @router.post(
            "/jobs/{name}/pause",
        )
        async def pause_job(
            name: str,
            workspace: str,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> TypedJobResponse:
            f"""Pause a job by name for the {service_name} microservice."""

            job_resp = (
                await client_from_platform(sdk, AsyncJobsClient).pause_job(name=name, workspace=workspace)
            ).data()
            return from_response(job_resp)

        @router.post(
            "/jobs/{name}/resume",
        )
        async def resume_job(
            name: str,
            workspace: str,
            sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> TypedJobResponse:
            f"""Resume a job by name for the {service_name} microservice."""

            job_resp = (
                await client_from_platform(sdk, AsyncJobsClient).resume_job(name=name, workspace=workspace)
            ).data()
            return from_response(job_resp)

        _stamp(pause_job, perm="pause", write=True)
        _stamp(resume_job, perm="resume", write=True)

    return router
