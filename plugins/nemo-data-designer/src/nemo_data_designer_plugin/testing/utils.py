# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import io
import json
import tempfile
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import unquote, urlparse

import click.testing
import data_designer.config as dd
import duckdb
import pandas as pd
import typer
import typer.testing
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer_nemo.nemotron_personas import get_file_path_for_locale, get_resource_name_for_locale
from nemo_data_designer_plugin.cli.main import DataDesignerCLI
from nemo_data_designer_plugin.functions.preview import PreviewFunction
from nemo_data_designer_plugin.jobs.create import CreateJob
from nemo_data_designer_plugin.jobs.spec import DataDesignerJobConfig
from nemo_data_designer_plugin.sdk.resources import DataDesignerResource
from nemo_data_designer_plugin.service import DataDesignerService
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.commands import add_function_commands, add_job_commands
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import PlatformJobResults
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.jobs.result_manager import ResultManager
from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.secrets.types import PlatformSecretCreateRequest
from nmp.core.files.service import FilesService
from nmp.core.inference_gateway.service import InferenceGatewayService
from nmp.core.jobs.service import JobsService
from nmp.core.models.service import ModelsService
from nmp.core.secrets.service import SecretsService
from nmp.platform_runner.plugin_adapter import NemoServiceAdapter
from nmp.testing import ClientContext, TaskResult, add_mock_provider, create_test_client, subprocess_job_executor_patch
from pydantic import SecretStr

WORKSPACE_NAME = "my-workspace"

ENABLED_MODEL_NAME = "nano-v3"

SECRET_NAME = "my-secret"
SECRET_RAW_VALUE = "abc123"

SEED_DATA = pd.DataFrame(
    data={
        "first_name": ["John", "Miles", "Bill"],
        "last_name": ["Coltrane", "Davis", "Evans"],
    }
)
FULL_NAME_EXPR = "{{ first_name }} + {{ last_name }}"
FULL_NAMES = {
    "John + Coltrane",
    "Miles + Davis",
    "Bill + Evans",
}
FILESET_NAME = "my-fileset"
FILE_PATH = "data.parquet"
FILESET_FILE_SEED_SOURCE_PATH = f"{WORKSPACE_NAME}/{FILESET_NAME}#{FILE_PATH}"

_MOCK_PREFIX = "igw-mock-"
_RAW_OPEN_PROVIDER_NAME = "open-provider"
_RAW_RESTRICTED_PROVIDER_NAME = "restricted-provider"
OPEN_PROVIDER_NAME = f"{_MOCK_PREFIX}{_RAW_OPEN_PROVIDER_NAME}"
RESTRICTED_PROVIDER_NAME = f"{_MOCK_PREFIX}{_RAW_RESTRICTED_PROVIDER_NAME}"

FAILING_RESULT_MANAGER_ERROR_MESSAGE = "This is a forced failure from the FailingResultManager"
FAILING_RESULT_MANAGER_MAX_SUCCESSFUL_CALLS = 2


def make_model_config(
    alias: str = "model-config-alias",
    model: str = ENABLED_MODEL_NAME,
    provider: str = OPEN_PROVIDER_NAME,
) -> dd.ModelConfig:
    return dd.ModelConfig(
        alias=alias,
        model=model,
        provider=provider,
        inference_parameters=dd.ChatCompletionInferenceParams(top_p=1),
    )


class MockHuggingFaceSeedReader(SeedReader[dd.HuggingFaceSeedSource]):
    _table_name = "df"

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        # The real dd.HuggingFaceSeedReader resolves the provided token here
        # and uses it to create a HF fsspec filesystem. The purpose of this
        # mock reader object is to skip calling out to real HF, but we *do*
        # want to test that secrets are resolved properly from within the
        # Data Designer library engine
        self.secret_resolver.resolve(self.source.token) if self.source.token else None

        conn = duckdb.connect()
        conn.register(self._table_name, SEED_DATA)
        return conn

    def get_dataset_uri(self) -> str:
        return self._table_name


@contextmanager
def mock_hf_seed_reader() -> Generator[None]:
    with patch("data_designer_nemo.context.HuggingFaceSeedReader", MockHuggingFaceSeedReader):
        yield


@contextmanager
def make_mock_client_context(workspace: str = WORKSPACE_NAME) -> Generator[ClientContext]:
    def _dd_service_factory() -> NemoServiceAdapter:
        return NemoServiceAdapter(DataDesignerService())

    with create_test_client(
        _dd_service_factory,
        FilesService,
        ModelsService,
        InferenceGatewayService,
        SecretsService,
        client_type=ClientContext,
        workspace=workspace,
        workspaces=[workspace],
    ) as client_context:
        yield client_context


@contextmanager
def setup_mock_providers(client_context: ClientContext) -> Generator[None]:
    add_mock_provider(
        sdk=client_context.sdk,
        workspace=client_context.sdk.workspace or WORKSPACE_NAME,
        name=_RAW_OPEN_PROVIDER_NAME,
    )
    add_mock_provider(
        sdk=client_context.sdk,
        workspace=client_context.sdk.workspace or WORKSPACE_NAME,
        name=_RAW_RESTRICTED_PROVIDER_NAME,
        enabled_models=[ENABLED_MODEL_NAME],
    )
    yield


@contextmanager
def setup_mock_secret(client_context: ClientContext) -> Generator[None]:
    secrets = client_from_platform(client_context.sdk, SecretsClient)
    secrets.create_secret(
        body=PlatformSecretCreateRequest(name=SECRET_NAME, value=SecretStr(SECRET_RAW_VALUE)),
        workspace=client_context.sdk.workspace or WORKSPACE_NAME,
    )
    yield


@contextmanager
def setup_mock_file(
    client_context: ClientContext,
    remote_path: str | None = None,
) -> Generator[None]:
    files = client_from_platform(client_context.sdk, FilesClient)
    files.create_fileset(
        body=CreateFilesetRequest(name=FILESET_NAME),
        workspace=client_context.sdk.workspace or WORKSPACE_NAME,
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmpfile:
        SEED_DATA.to_parquet(tmpfile.name, index=False)
        files.upload_file(
            name=FILESET_NAME,
            workspace=client_context.sdk.workspace or WORKSPACE_NAME,
            path=remote_path or FILE_PATH,
            content=Path(tmpfile.name).read_bytes(),
        )
    yield


@contextmanager
def setup_mock_nemotron_personas_data(
    client_context: ClientContext,
    persona_data: pd.DataFrame,
) -> Generator[None]:
    """Create the expected personas fileset for the en_US locale with dummy data.

    Note that which PersonSamplerParams filtering criteria will be supported depends on
    the data you provide. For example, if you want to filter on `age_range`, be sure to
    include an `age` column in your data.
    There is a minimal set of columns required by the library, and this function does validate
    that that minimal set is included, but that set is not equal to (and smaller than) the set
    of columns required for all PersonSamplerParams filtering fields to work.

    """

    _create_nemotron_personas_fileset(client_context.sdk, persona_data)
    yield


def _create_nemotron_personas_fileset(sdk: NeMoPlatform, persona_data: pd.DataFrame) -> None:
    fileset_name = get_resource_name_for_locale("en_US")
    files = client_from_platform(sdk, FilesClient)
    files.create_fileset(body=CreateFilesetRequest(name=fileset_name), workspace="system")
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmpfile:
        persona_data.to_parquet(tmpfile.name, index=False)
        files.upload_file(
            workspace="system",
            name=fileset_name,
            path=get_file_path_for_locale("en_US"),
            content=Path(tmpfile.name).read_bytes(),
        )


async def compile_create_job(
    original_spec: DataDesignerJobConfig,
    workspace: str = WORKSPACE_NAME,
    sdk: AsyncNeMoPlatform | None = None,
) -> PlatformJobSpec:
    sdk = sdk or AsyncMock(spec=AsyncNeMoPlatform)
    entity_client = Mock()
    job = CreateJob()
    # This helper exercises the plugin-service compilation path, where
    # `to_spec` runs before a platform job exists.
    step_config = await job.to_spec(
        original_spec,
        workspace=workspace,
        entity_client=entity_client,
        async_sdk=sdk,
        is_local=False,
    )
    return await job.compile(
        workspace=workspace,
        spec=step_config,
        entity_client=entity_client,
        job_name=None,
        async_sdk=sdk,
    )


def make_dd_client(client_context: ClientContext) -> DataDesignerResource:
    return DataDesignerResource(client_context.sdk)


def _make_data_designer_cli_app() -> typer.Typer:
    cli = DataDesignerCLI()
    app = cli.get_cli()
    add_function_commands(app, {"preview": PreviewFunction}, cli=cli)
    add_job_commands(app, {"create": CreateJob}, cli=cli)
    return app


@dataclass
class DataDesignerCLIState:
    sdk: NeMoPlatform
    async_sdk: AsyncNeMoPlatform
    overrides: dict[str, Any]

    def get_client(self) -> NeMoPlatform:
        return self.sdk

    def get_async_client(self) -> AsyncNeMoPlatform:
        return self.async_sdk


def _make_data_designer_cli_state(
    client_context: ClientContext,
    *,
    output_format: str | None = None,
) -> DataDesignerCLIState:
    # Mirrors what `nemo --output-format json` would do at the top-level callback.
    # The plugin's test app doesn't mount the real top-level callback, so we set this directly.
    overrides = {"output_format": output_format} if output_format is not None else {}
    return DataDesignerCLIState(
        sdk=client_context.sdk,
        async_sdk=client_context.async_sdk,
        overrides=overrides,
    )


def invoke_cli(
    command: list[str],
    client_context: ClientContext | None = None,
    output_format: Literal["json"] | None = None,
) -> click.testing.Result:
    runner = typer.testing.CliRunner()
    app = _make_data_designer_cli_app()

    cli_state = None
    if client_context is not None:
        cli_state = _make_data_designer_cli_state(client_context, output_format=output_format)

    return runner.invoke(app, command, obj=cli_state)


def write_config_file(tmp_path: Path, source: str, *, name: str = "data_designer_config.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def find_single_preview_results_dir(path: Path) -> Path:
    matches = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith("preview_results_"))
    if len(matches) != 1:
        raise AssertionError(f"Expected one preview results directory in {path}, found {len(matches)}: {matches}")
    return matches[0]


def read_saved_preview_dataset(results_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(results_dir / "dataset.parquet")


def read_file_url(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise ValueError(f"Expected a file:// URL, got {url!r}")
    return Path(unquote(parsed.path))


def parse_cli_json_object(output: str) -> dict[str, Any]:
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise AssertionError(f"No JSON object found in CLI output:\n{output}")
    parsed = json.loads(output[start : end + 1])
    if not isinstance(parsed, dict):
        raise AssertionError(f"Expected a JSON object in CLI output, got {type(parsed).__name__}")
    return parsed


def _normalize_job_config(job_config: Any) -> dict[str, Any]:
    if isinstance(job_config, dict):
        return job_config
    return job_config.model_dump(mode="json")


@dataclass
class CreateJobTestContext:
    sdk: NeMoPlatform
    async_sdk: AsyncNeMoPlatform
    config: dict[str, Any]
    job_ctx: JobContext

    def run_task(self) -> TaskResult:
        """Compatibility shim for older task-harness tests."""
        return self.run_job()

    def run_job(self) -> TaskResult:
        def _execute_job() -> TaskResult:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            exit_code = 0
            exception = None

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = CreateJob().run(self.config, ctx=self.job_ctx, sdk=self.sdk)
                    exit_code = result["exit_code"]
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 1
            except Exception as e:
                exception = e
                exit_code = 1

            return TaskResult(
                exit_code=exit_code,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                exception=exception,
            )

        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False

        if has_loop:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(_execute_job).result()

        return _execute_job()


@asynccontextmanager
async def task_context(
    job_config: PlatformJobSpec | dict[str, Any], job_name: str
) -> AsyncGenerator[CreateJobTestContext]:
    class _TestDataDesignerService(NemoServiceAdapter):
        def __init__(self) -> None:
            super().__init__(DataDesignerService())

    job_config_dict = _normalize_job_config(job_config)
    step_config = job_config_dict["steps"][0]["config"]

    with tempfile.TemporaryDirectory() as storage_tmpdir:
        storage_root = Path(storage_tmpdir)
        ephemeral = storage_root / "ephemeral"
        persistent = storage_root / "persistent"
        ephemeral.mkdir()
        persistent.mkdir()

        with (
            subprocess_job_executor_patch(),
            create_test_client(
                _TestDataDesignerService,
                FilesService,
                JobsService,
                client_type=ClientContext,
                workspaces=["default"],
                workspace="default",
            ) as client_context,
        ):
            jobs_client = client_from_platform(client_context.sdk, JobsClient)
            job = jobs_client.create_job(
                workspace="default",
                body=CreatePlatformJobRequest(
                    name=job_name,
                    source="data-designer",
                    # Store the canonical DataDesignerStepConfig as the job's spec so that
                    # downstream Data Designer routes (e.g. ``GET /jobs/create/{name}``,
                    # which deserializes the stored spec back through the schema) succeed.
                    spec=step_config if isinstance(step_config, dict) else step_config.model_dump(),
                    platform_spec=job_config_dict,  # ty:ignore[invalid-argument-type]
                ),
            ).data()
            job_ctx = JobContext(
                workspace="default",
                storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
                results=PlatformJobResults(
                    job_name=job_name,
                    workspace="default",
                    client=client_from_platform(client_context.sdk, NemoClient),
                ),
                job_id=job.id,
            )
            yield CreateJobTestContext(
                sdk=client_context.sdk,
                async_sdk=client_context.async_sdk,
                config=step_config,
                job_ctx=job_ctx,
            )


class FailingResultManager(ResultManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_count = 0
        self._fail_on_attempt_number = FAILING_RESULT_MANAGER_MAX_SUCCESSFUL_CALLS + 1

    def create_result(self, *args, **kwargs):
        self._call_count += 1
        if self._call_count >= self._fail_on_attempt_number:
            raise RuntimeError(FAILING_RESULT_MANAGER_ERROR_MESSAGE)
        return super().create_result(*args, **kwargs)
