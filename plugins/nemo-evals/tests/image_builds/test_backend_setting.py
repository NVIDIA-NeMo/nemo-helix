# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Builder backend evals expects a deployment's task images from: the one ``builder.backend`` names."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from nemo_builder_plugin.backends import BACKENDS
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import IN_CLUSTER_BACKEND
from nemo_evals.api.dependencies import get_image_build_service
from nemo_evals.config import expected_builder_backend
from nemo_evals.image_builds.service import ImageBuildService
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.config import Configuration
from pydantic import ValidationError

REMOTE_BACKEND = "remote"


def _builder_reads(monkeypatch: pytest.MonkeyPatch, section: dict[str, Any]) -> None:
    """Have Builder's config class read ``section`` as the ``builder`` section, checked on each read."""

    def get(cls: type[BuilderConfig]) -> BuilderConfig:
        return BuilderConfig.model_validate(section)

    monkeypatch.setattr(BuilderConfig, "get", classmethod(get))


def _builder_reads_file(monkeypatch: pytest.MonkeyPatch, config_file: Path) -> None:
    """Have Builder's config class read its section from ``config_file``, as the platform reads its own file."""

    def get(cls: type[BuilderConfig]) -> BuilderConfig:
        return Configuration.get_service_config_from_file(str(config_file), cls)

    monkeypatch.setattr(BuilderConfig, "get", classmethod(get))


def test_a_deployment_that_names_no_backend_expects_the_in_cluster_one(monkeypatch) -> None:
    """Builder's default is the in-cluster backend, by the name Builder records on a row it builds in this cluster."""
    _builder_reads(monkeypatch, {})

    assert BuilderConfig.model_fields["backend"].default == IN_CLUSTER_BACKEND
    assert expected_builder_backend() == IN_CLUSTER_BACKEND


@pytest.mark.parametrize("backend", [IN_CLUSTER_BACKEND, REMOTE_BACKEND])
def test_evals_expects_the_backend_builder_is_configured_with(monkeypatch, backend: str) -> None:
    """An operator switches ``builder.backend`` and nothing under ``evals``."""
    _builder_reads(monkeypatch, {"backend": backend})

    assert backend in BACKENDS
    assert expected_builder_backend() == backend


@pytest.mark.parametrize(
    "section",
    [{"backend": REMOTE_BACKEND, "remote": {"repositry": "team/images"}}, {"registry": "bot:t0ken@registry.test"}],
    ids=["a-misspelt-key", "a-value-builder-refuses"],
)
def test_a_builder_section_that_fails_its_checks_gives_no_backend(monkeypatch, caplog, section: dict[str, Any]) -> None:
    """Builder refuses every submit then, and says so: evals fails no request over it, and logs none of it."""
    _builder_reads(monkeypatch, section)
    with pytest.raises(ValidationError) as refused:
        BuilderConfig.get()
    # The error quotes the value it refused, which here holds a login.
    assert ("t0ken" in str(refused.value)) == ("registry" in section)
    caplog.set_level(logging.DEBUG)

    assert expected_builder_backend() is None
    assert not caplog.records
    assert "t0ken" not in caplog.text


@pytest.mark.parametrize("text", ["builder: remote\n", "builder:\n"], ids=["not-a-mapping", "left-empty"])
def test_a_builder_section_that_cant_be_read_at_all_gives_no_backend(tmp_path: Path, monkeypatch, text: str) -> None:
    """Read as the platform reads its file. The loader's own error is not the one a setting's check raises."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(text)
    _builder_reads_file(monkeypatch, config_file)
    with pytest.raises(ValueError, match="must be a mapping") as unreadable:
        BuilderConfig.get()
    assert not isinstance(unreadable.value, ValidationError)

    assert expected_builder_backend() is None


@pytest.mark.parametrize("backend", ["Execution", "remot", ""], ids=["another-case", "misspelt", "empty"])
def test_a_backend_builder_doesnt_have_gives_no_backend(monkeypatch, backend: str) -> None:
    """The section passes its checks with any name, and Builder then refuses every submit for one it doesn't have.

    Expecting that name would turn each task's ready build down for an attempt that can have nothing built.
    """
    _builder_reads(monkeypatch, {"backend": backend})

    assert BuilderConfig.get().backend == backend
    assert expected_builder_backend() is None


def test_one_platform_configuration_that_switches_builder_switches_evals_too(tmp_path: Path, monkeypatch) -> None:
    """The file names a backend under ``builder`` and holds nothing under ``evals``."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("builder:\n  backend: remote\n")
    monkeypatch.delenv("NEMO_BUILDER_BACKEND", raising=False)
    _builder_reads_file(monkeypatch, config_file)

    assert expected_builder_backend() == REMOTE_BACKEND


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
    _builder_reads(monkeypatch, {"backend": backend})

    assert _routes_service(entity_store).builder_backend == backend


def test_the_routes_service_of_a_deployment_that_names_no_backend_takes_the_in_cluster_one(
    entity_store, monkeypatch
) -> None:
    _builder_reads(monkeypatch, {})

    assert _routes_service(entity_store).builder_backend == IN_CLUSTER_BACKEND


def test_the_routes_service_is_given_no_backend_while_builders_cant_be_told(entity_store, monkeypatch) -> None:
    """Not the in-cluster one: the service then turns down no ready build for the backend that built it."""
    _builder_reads(monkeypatch, {"backend": "remot"})

    assert _routes_service(entity_store).builder_backend is None
