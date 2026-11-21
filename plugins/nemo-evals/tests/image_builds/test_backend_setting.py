# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evals setting that names the Builder backend a deployment's task images must come from."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import IN_CLUSTER_BACKEND
from nemo_evals.api.dependencies import get_image_build_service
from nemo_evals.config import EvaluatorConfig, ImageBuildsConfig
from nemo_evals.image_builds.service import ImageBuildService
from nemo_helix_plugin.client.client import AsyncNemoClient
from pydantic import ValidationError

REMOTE = {"image_builds": {"builder_backend": "remote"}}


def test_a_deployment_that_sets_neither_key_has_evals_and_builder_agree(monkeypatch) -> None:
    """Both default to the in-cluster backend, by the name Builder records on a row it builds in this cluster."""
    monkeypatch.delenv("NEMO_EVALS_IMAGE_BUILDS", raising=False)

    assert EvaluatorConfig().image_builds.builder_backend == IN_CLUSTER_BACKEND
    assert BuilderConfig.model_fields["backend"].default == IN_CLUSTER_BACKEND


def test_the_backend_is_set_under_image_builds_in_the_evals_section() -> None:
    assert EvaluatorConfig.model_validate(REMOTE).image_builds.builder_backend == "remote"


def test_the_backend_is_read_from_the_environment_as_one_json_value(monkeypatch) -> None:
    monkeypatch.setenv("NEMO_EVALS_IMAGE_BUILDS", '{"builder_backend":"remote"}')

    assert EvaluatorConfig().image_builds.builder_backend == "remote"


def test_an_unknown_image_builds_key_is_refused() -> None:
    """A misspelt key must not leave a deployment on the default without saying so."""
    with pytest.raises(ValidationError, match="backend"):
        EvaluatorConfig.model_validate({"image_builds": {"backend": "remote"}})


@pytest.mark.parametrize("backend", ["", "x" * 65], ids=["empty", "longer-than-a-row-records"])
def test_a_backend_no_builder_row_could_name_is_refused(backend: str) -> None:
    with pytest.raises(ValidationError, match="builder_backend"):
        ImageBuildsConfig(builder_backend=backend)


async def _authorize(method: str, path: str) -> None:
    pass


def _routes_service(entity_store) -> ImageBuildService:
    """The service the image build routes are given, as FastAPI resolves it for a request in ``default``."""
    return get_image_build_service(
        "default", entity_store, AsyncNemoClient(base_url="http://platform.test"), _authorize
    )


def test_the_routes_service_takes_the_backend_the_deployment_names(entity_store, monkeypatch) -> None:
    monkeypatch.setattr("nemo_evals.api.dependencies.get_config", lambda: EvaluatorConfig.model_validate(REMOTE))

    assert _routes_service(entity_store).builder_backend == "remote"


def test_the_routes_service_of_a_deployment_that_names_none_takes_the_in_cluster_backend(
    entity_store, monkeypatch
) -> None:
    monkeypatch.setattr("nemo_evals.api.dependencies.get_config", lambda: EvaluatorConfig.model_validate({}))

    assert _routes_service(entity_store).builder_backend == IN_CLUSTER_BACKEND
