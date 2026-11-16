# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError

import pytest
from nhx.platform_runner.version import _resolve_version, get_revision


def _package_version(value: str | None):
    def _version(_name: str) -> str:
        if value is None:
            raise PackageNotFoundError("nemo-helix")
        return value

    return _version


@pytest.mark.parametrize("package_version", ["0.0.0", "1.2.3", None])
@pytest.mark.parametrize("env_version", ["0.8.1", "v0.8.1", "0.9.0rc1"])
def test_resolve_version_prefers_env_over_package(monkeypatch, package_version, env_version):
    """Images stamp the release into NHX_PLATFORM_VERSION; a tag-less wheel build must not override it."""
    monkeypatch.setattr("nhx.platform_runner.version.version", _package_version(package_version))
    monkeypatch.setenv("NHX_PLATFORM_VERSION", f" {env_version} ")
    assert _resolve_version() == env_version


@pytest.mark.parametrize(
    "env_version",
    [None, "", "dev", "0.0.0", "local", "d24f2a7718ecd7d9cf388edb749e562f4ce514bc", "1234567"],
)
def test_resolve_version_uses_package_when_env_is_not_a_release(monkeypatch, env_version):
    """Bake tags default to "local" and to the commit SHA in CI; neither is a platform version."""
    monkeypatch.setattr("nhx.platform_runner.version.version", _package_version("1.2.3"))
    if env_version is None:
        monkeypatch.delenv("NHX_PLATFORM_VERSION", raising=False)
    else:
        monkeypatch.setenv("NHX_PLATFORM_VERSION", env_version)
    assert _resolve_version() == "1.2.3"


@pytest.mark.parametrize("package_version", ["0.0.0", "", None])
@pytest.mark.parametrize("env_version", [None, "dev", "local"])
def test_resolve_version_reports_dev_when_nothing_is_stamped(monkeypatch, package_version, env_version):
    """The dynamic-versioning fallback 0.0.0 reads as a real release, so it is reported as dev."""
    monkeypatch.setattr("nhx.platform_runner.version.version", _package_version(package_version))
    if env_version is None:
        monkeypatch.delenv("NHX_PLATFORM_VERSION", raising=False)
    else:
        monkeypatch.setenv("NHX_PLATFORM_VERSION", env_version)
    assert _resolve_version() == "dev"


def test_get_revision(monkeypatch):
    monkeypatch.setenv("NHX_CODE_REVISION", "abc123")
    assert get_revision() == "abc123"
    monkeypatch.delenv("NHX_CODE_REVISION", raising=False)
    assert get_revision() == "dev"
