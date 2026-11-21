# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Builder backend evals expects a deployment's task images from, and the evals key that overrides it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import IN_CLUSTER_BACKEND
from nemo_evals.api.dependencies import get_image_build_service
from nemo_evals.config import EvaluatorConfig, ImageBuildsConfig, expected_builder_backend
from nemo_evals.image_builds.service import ImageBuildService
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.config import Configuration
from pydantic import ValidationError

REMOTE_BACKEND = "remote"
REMOTE = {"image_builds": {"builder_backend": REMOTE_BACKEND}}


def _evals(builder_backend: str | None = None) -> EvaluatorConfig:
    """The evals section of a deployment that sets the key to ``builder_backend``, or leaves it out."""
    section = {} if builder_backend is None else {"image_builds": {"builder_backend": builder_backend}}
    return EvaluatorConfig.model_validate(section)


def _builder_reads(monkeypatch: pytest.MonkeyPatch, section: dict[str, Any]) -> None:
    """Have Builder's config class read ``section`` as the ``builder`` section, checked on each read."""

    def get(cls: type[BuilderConfig]) -> BuilderConfig:
        return BuilderConfig.model_validate(section)

    monkeypatch.setattr(BuilderConfig, "get", classmethod(get))


def test_the_key_is_unset_unless_a_deployment_sets_it(monkeypatch) -> None:
    monkeypatch.delenv("NEMO_EVALS_IMAGE_BUILDS", raising=False)

    assert EvaluatorConfig().image_builds.builder_backend is None
    for section in ({}, {"builder_backend": None}):
        assert EvaluatorConfig.model_validate({"image_builds": section}).image_builds.builder_backend is None


def test_a_deployment_that_sets_neither_key_expects_the_in_cluster_backend(monkeypatch) -> None:
    """Builder's default is the in-cluster backend, by the name Builder records on a row it builds in this cluster."""
    _builder_reads(monkeypatch, {})

    assert BuilderConfig.model_fields["backend"].default == IN_CLUSTER_BACKEND
    assert expected_builder_backend(_evals()) == IN_CLUSTER_BACKEND


@pytest.mark.parametrize("backend", [IN_CLUSTER_BACKEND, REMOTE_BACKEND])
def test_with_the_key_unset_evals_expects_the_backend_builder_is_configured_with(monkeypatch, backend: str) -> None:
    """An operator switches ``builder.backend`` and nothing under ``evals``."""
    _builder_reads(monkeypatch, {"backend": backend})

    assert expected_builder_backend(_evals()) == backend


@pytest.mark.parametrize(
    ("key", "builders"),
    [(REMOTE_BACKEND, IN_CLUSTER_BACKEND), (IN_CLUSTER_BACKEND, REMOTE_BACKEND)],
    ids=["remote-over-in-cluster", "in-cluster-over-remote"],
)
def test_the_key_overrides_the_backend_builder_is_configured_with(monkeypatch, key: str, builders: str) -> None:
    """For a deployment whose evals API reads another configuration than its Builder does."""
    _builder_reads(monkeypatch, {"backend": builders})

    assert expected_builder_backend(_evals(key)) == key


def test_with_the_key_set_builders_configuration_is_not_read(monkeypatch) -> None:
    """The configuration such a deployment's evals API reads may hold no Builder section that can be read."""

    def get(cls: type[BuilderConfig]) -> BuilderConfig:
        pytest.fail("Builder's configuration was read")

    monkeypatch.setattr(BuilderConfig, "get", classmethod(get))

    assert expected_builder_backend(_evals(REMOTE_BACKEND)) == REMOTE_BACKEND


@pytest.mark.parametrize(
    "section",
    [{"backend": REMOTE_BACKEND, "remote": {"repositry": "team/images"}}, {"registry": "bot:t0ken@registry.test"}],
    ids=["a-misspelt-key", "a-value-builder-refuses"],
)
def test_a_builder_section_that_cant_be_read_leaves_evals_on_the_in_cluster_backend(
    monkeypatch, caplog, section: dict[str, Any]
) -> None:
    """Builder refuses every submit then, and says so: evals fails no request over it, and logs none of it."""
    _builder_reads(monkeypatch, section)
    with pytest.raises(ValidationError):
        BuilderConfig.get()

    assert expected_builder_backend(_evals()) == IN_CLUSTER_BACKEND
    assert not caplog.records


def test_one_platform_configuration_that_switches_builder_alone_switches_evals_too(tmp_path: Path, monkeypatch) -> None:
    """Both sections are read from one file, as the platform reads it: it names a backend under ``builder`` only."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("builder:\n  backend: remote\n")
    monkeypatch.delenv("NEMO_EVALS_IMAGE_BUILDS", raising=False)
    monkeypatch.delenv("NEMO_BUILDER_BACKEND", raising=False)

    def get(cls: type[BuilderConfig]) -> BuilderConfig:
        return Configuration.get_service_config_from_file(str(config_file), cls)

    monkeypatch.setattr(BuilderConfig, "get", classmethod(get))
    evals = Configuration.get_service_config_from_file(str(config_file), EvaluatorConfig)

    assert evals.image_builds.builder_backend is None
    assert expected_builder_backend(evals) == REMOTE_BACKEND


def test_the_backend_is_set_under_image_builds_in_the_evals_section() -> None:
    assert EvaluatorConfig.model_validate(REMOTE).image_builds.builder_backend == REMOTE_BACKEND


def test_the_backend_is_read_from_the_environment_as_one_json_value(monkeypatch) -> None:
    monkeypatch.setenv("NEMO_EVALS_IMAGE_BUILDS", '{"builder_backend":"remote"}')

    assert EvaluatorConfig().image_builds.builder_backend == REMOTE_BACKEND


def test_an_unknown_image_builds_key_is_refused() -> None:
    """A misspelt key must not leave a deployment following Builder's backend without saying so."""
    with pytest.raises(ValidationError, match="backend"):
        EvaluatorConfig.model_validate({"image_builds": {"backend": "remote"}})


@pytest.mark.parametrize("backend", ["", "x" * 65], ids=["empty", "longer-than-a-row-records"])
def test_a_backend_no_builder_row_could_name_is_refused(backend: str) -> None:
    with pytest.raises(ValidationError, match="builder_backend"):
        ImageBuildsConfig(builder_backend=backend)


def test_a_backend_as_long_as_a_builder_row_records_is_taken() -> None:
    assert ImageBuildsConfig(builder_backend="x" * 64).builder_backend == "x" * 64


async def _authorize(method: str, path: str) -> None:
    pass


def _routes_service(entity_store) -> ImageBuildService:
    """The service the image build routes are given, as FastAPI resolves it for a request in ``default``."""
    return get_image_build_service(
        "default", entity_store, AsyncNemoClient(base_url="http://platform.test"), _authorize
    )


@pytest.mark.parametrize("backend", [IN_CLUSTER_BACKEND, REMOTE_BACKEND])
def test_the_routes_service_takes_the_backend_builder_is_configured_with(
    entity_store, monkeypatch, backend: str
) -> None:
    monkeypatch.setattr("nemo_evals.api.dependencies.get_config", _evals)
    _builder_reads(monkeypatch, {"backend": backend})

    assert _routes_service(entity_store).builder_backend == backend


@pytest.mark.parametrize(
    ("key", "builders"),
    [(REMOTE_BACKEND, IN_CLUSTER_BACKEND), (IN_CLUSTER_BACKEND, REMOTE_BACKEND)],
    ids=["remote-over-in-cluster", "in-cluster-over-remote"],
)
def test_the_routes_service_takes_the_backend_the_key_names_over_builders(
    entity_store, monkeypatch, key: str, builders: str
) -> None:
    monkeypatch.setattr("nemo_evals.api.dependencies.get_config", lambda: _evals(key))
    _builder_reads(monkeypatch, {"backend": builders})

    assert _routes_service(entity_store).builder_backend == key


def test_the_routes_service_of_a_deployment_that_names_no_backend_takes_the_in_cluster_one(
    entity_store, monkeypatch
) -> None:
    monkeypatch.setattr("nemo_evals.api.dependencies.get_config", _evals)
    _builder_reads(monkeypatch, {})

    assert _routes_service(entity_store).builder_backend == IN_CLUSTER_BACKEND
