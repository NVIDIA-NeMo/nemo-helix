# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fileset file types are validated before an Anonymizer job is accepted."""

from collections.abc import Generator

import data_designer.config as dd
import pytest
from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Redact
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec
from nemo_anonymizer_plugin.app.task_config import AnonymizerRequest
from nemo_anonymizer_plugin.sdk.errors import AnonymizerConfigValidationError
from nemo_anonymizer_plugin.sdk.resources import AnonymizerResource
from nemo_anonymizer_plugin.service import AnonymizerService
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nhx.core.files.service import FilesService
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.core.jobs.service import JobsService
from nhx.core.models.service import ModelsService
from nhx.core.secrets.service import SecretsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, add_mock_provider, create_test_client, subprocess_job_executor_patch

pytestmark = pytest.mark.integration

WORKSPACE = "default"


def _anonymizer_service_factory() -> NemoServiceAdapter:
    return NemoServiceAdapter(AnonymizerService())


@pytest.fixture
def client_context() -> Generator[ClientContext]:
    # Mount services only: submission must not execute a task or run inference.
    with create_test_client(
        _anonymizer_service_factory,
        FilesService,
        JobsService,
        ModelsService,
        InferenceGatewayService,
        SecretsService,
        client_type=ClientContext,
        workspace=WORKSPACE,
        workspaces=[WORKSPACE],
    ) as ctx:
        add_mock_provider(ctx.client, workspace=WORKSPACE, name="detector")
        files = FilesClient.from_client(ctx.client)
        files.create_fileset(workspace=WORKSPACE, body=CreateFilesetRequest(name="inputs"))
        files.upload_file(
            workspace=WORKSPACE,
            name="inputs",
            path="data/records.csv",
            content=b"text\nHello world\n",
        )
        files.upload_file(
            workspace=WORKSPACE,
            name="inputs",
            path="data/records.JSON",
            content=b'[{"text": "Hello world"}]',
        )
        with subprocess_job_executor_patch():
            yield ctx


def _request(source: str) -> AnonymizerRequest:
    return AnonymizerRequest(
        config=AnonymizerConfig(replace=Redact()),
        data=AnonymizerInputSpec(source=source),
        model_configs=[dd.ModelConfig(alias="detector", model="detector", provider="igw-mock-detector")],
    )


def test_csv_fileset_reference_is_accepted(client_context: ClientContext) -> None:
    anonymizer = AnonymizerResource(client_context.client)
    source = "inputs#data/records.csv"

    handle = anonymizer.run(_request(source))
    job = handle.get_job()

    assert job["id"]
    assert job["name"]
    assert job["spec"]["request"]["data"]["source"] == source
    assert anonymizer.get_job_resource(job["name"]).get_job()["id"] == job["id"]


def test_unsupported_fileset_type_is_rejected_at_submission(client_context: ClientContext) -> None:
    anonymizer = AnonymizerResource(client_context.client)
    source = "inputs#data/records.JSON"

    with pytest.raises(AnonymizerConfigValidationError) as exc_info:
        anonymizer.run(_request(source))

    assert "Config validation failed!" in str(exc_info.value)
    assert f"Fileset input source {source!r} must point to a .csv or .parquet file, got '.json'." in str(exc_info.value)
