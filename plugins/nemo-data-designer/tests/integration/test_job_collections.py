# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each data-designer job collection lists and serves only its own jobs.

The four collections (create, retrieval-generate, retrieval-prepare, retrieval-run) once shared one
Jobs ``source``. Each list route filters on ``source`` and renders every match with its own spec
schema, so a single job of another type made the list return 500. Create keeps that shared source;
each retrieval collection now has its own.

Jobs are seeded straight into the core Jobs service with the ``source`` and stored spec each
collection writes, so no job compiles or runs.
"""

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from typing import Any

import data_designer.config as dd
import pytest
import time_machine
from fastapi.testclient import TestClient
from nemo_data_designer_plugin.jobs.retrieval_spec import (
    RetrievalGenerateJobConfig,
    RetrievalGenerateStepConfig,
    RetrievalPrepareJobConfig,
    RetrievalPrepareStepConfig,
    RetrievalRunJobConfig,
)
from nemo_data_designer_plugin.service import (
    CREATE_JOB_SOURCE,
    RETRIEVAL_GENERATE_JOB_SOURCE,
    RETRIEVAL_PREPARE_JOB_SOURCE,
    RETRIEVAL_RUN_JOB_SOURCE,
    DataDesignerService,
)
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.core.files.service import FilesService
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.service import JobsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client

pytestmark = pytest.mark.integration

WORKSPACE = "default"

_PLATFORM_SPEC = HelixJobSpec.model_validate(
    {
        "steps": [
            {
                "name": "step1",
                "executor": {
                    "provider": "cpu",
                    "profile": "default",
                    "container": {"image": "test", "command": ["true"]},
                },
            }
        ]
    }
)


def _create_spec() -> dict[str, Any]:
    builder = dd.DataDesignerConfigBuilder()
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="category",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a", "b"]),
        )
    )
    return {
        "job_config": {"num_records": 3, "config": builder.build().model_dump(mode="json")},
        "model_providers": [],
        "model_configs": [],
    }


def _generate_config() -> RetrievalGenerateJobConfig:
    return RetrievalGenerateJobConfig.model_validate(
        {
            "corpus": "default/docs",
            "provider": "default/nvidia-build",
            "artifact_extraction_model": "nvidia/nemotron-3-nano-30b-a3b",
            "qa_generation_model": "nvidia/nemotron-3-nano-30b-a3b",
            "quality_judge_model": "nvidia/nemotron-3-nano-30b-a3b",
            "embed_model": "nvidia/nemotron-3-embed-1b",
        }
    )


def _retrieval_generate_spec() -> dict[str, Any]:
    return RetrievalGenerateStepConfig(
        job_config=_generate_config(),
        model_providers=[],
        chat_provider_name="chat",
        embed_provider_name="embed",
    ).model_dump(mode="json")


def _retrieval_prepare_spec() -> dict[str, Any]:
    return RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(sdg_input="default/sdg-output"),
    ).model_dump(mode="json")


def _retrieval_run_spec() -> dict[str, Any]:
    return RetrievalRunJobConfig(generate=_generate_config()).model_dump(mode="json")


#: (collection path segment, source, job name, stored spec) for one job of each type.
_SEEDED = [
    ("create", CREATE_JOB_SOURCE, "create-job", _create_spec()),
    ("retrieval-generate", RETRIEVAL_GENERATE_JOB_SOURCE, "generate-job", _retrieval_generate_spec()),
    ("retrieval-prepare", RETRIEVAL_PREPARE_JOB_SOURCE, "prepare-job", _retrieval_prepare_spec()),
    ("retrieval-run", RETRIEVAL_RUN_JOB_SOURCE, "run-job", _retrieval_run_spec()),
]


def _dd_service_factory() -> NemoServiceAdapter:
    return NemoServiceAdapter(DataDesignerService())


def _seed(ctx: ClientContext, *, name: str, source: str, spec: dict[str, Any]) -> None:
    JobsClient.from_client(ctx.client).create_job(
        workspace=WORKSPACE,
        body=CreateHelixJobRequest(name=name, source=source, spec=spec, platform_spec=_PLATFORM_SPEC),
    ).data()


def _list(http: TestClient, collection: str) -> Any:
    return http.get(f"/apis/data-designer/v2/workspaces/{WORKSPACE}/jobs/{collection}")


def _list_names(http: TestClient, collection: str) -> set[str]:
    response = _list(http, collection)
    assert response.status_code == 200, response.text
    return {job["name"] for job in response.json()["data"]}


@pytest.fixture
def ctx() -> Generator[ClientContext]:
    with create_test_client(
        _dd_service_factory,
        FilesService,
        JobsService,
        client_type=ClientContext,
        workspace=WORKSPACE,
        workspaces=[WORKSPACE],
    ) as client_context:
        for _collection, source, name, spec in _SEEDED:
            _seed(client_context, name=name, source=source, spec=spec)
        yield client_context


def test_collections_have_distinct_sources() -> None:
    sources = [source for _collection, source, *_ in _SEEDED]
    assert len(set(sources)) == len(sources)


def test_create_keeps_the_pre_split_source() -> None:
    """Existing create jobs stay listed only while create keeps the source they were stored under."""
    assert CREATE_JOB_SOURCE == "nemo-data-designer-plugin"


@pytest.mark.parametrize(("collection", "name"), [(collection, name) for collection, _, name, _ in _SEEDED])
def test_list_returns_only_own_jobs_in_mixed_workspace(ctx: ClientContext, collection: str, name: str) -> None:
    assert _list_names(ctx.test_client, collection) == {name}


def test_retrieval_job_stored_before_split_breaks_create_list_until_deleted(ctx: ClientContext) -> None:
    """A retrieval job created before the split still carries the create source.

    It drops out of its own retrieval list, and the create list returns 500 rendering it as a create
    spec. Cancelling and deleting it through the core Jobs API is the cleanup.
    """
    _seed(ctx, name="legacy-generate-job", source=CREATE_JOB_SOURCE, spec=_retrieval_generate_spec())

    assert _list_names(ctx.test_client, "retrieval-generate") == {"generate-job"}
    assert _list(ctx.test_client, "create").status_code == 500

    core_jobs = JobsClient.from_client(ctx.client)
    core_jobs.cancel_job(name="legacy-generate-job", workspace=WORKSPACE)
    core_jobs.delete_job(name="legacy-generate-job", workspace=WORKSPACE)

    # DELETE accepts cleanup; the existing reconciler must remove the Job first.
    with time_machine.travel(datetime.now(timezone.utc) + timedelta(hours=1), tick=False):
        JobReconciler(BackendRegistry({}), ctx.client)._cleanup_provisional_jobs()
    assert _list_names(ctx.test_client, "create") == {"create-job"}
