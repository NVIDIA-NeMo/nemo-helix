# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.harbor_runners import (
    UnsupportedHarborVersion,
    candidate_harbor_versions,
    qualification_catalog,
    resolve_harbor_runner,
    supported_harbor_versions,
)

CATALOG = qualification_catalog()
DEFAULT_VERSION = CATALOG["default_version"]


@pytest.mark.parametrize("selector", [None, "default", "stable", DEFAULT_VERSION])
def test_default_and_stable_resolve_to_exact_runner(selector: str | None) -> None:
    runner = resolve_harbor_runner(selector)

    assert runner.version == DEFAULT_VERSION
    assert runner.harbor_dir == f"/opt/harbor/{DEFAULT_VERSION}"
    assert runner.sandbox_k8s_version == CATALOG["sandbox_k8s"]["version"]
    assert runner.adapter_version == CATALOG["adapter"]["version"]


def test_default_aliases_are_explicit_stable_pins() -> None:
    assert CATALOG["aliases"]["default"] == DEFAULT_VERSION
    assert CATALOG["aliases"]["stable"] == DEFAULT_VERSION
    assert CATALOG["releases"][DEFAULT_VERSION]["selectable"] is True


def test_catalog_supports_curated_compatibility_lines() -> None:
    assert supported_harbor_versions() == tuple(
        version for version, release in CATALOG["releases"].items() if release["selectable"]
    )
    assert resolve_harbor_runner("0.6.3").harbor_dir == "/opt/harbor/0.6.3"


def test_package_compatible_candidates_are_not_selectable() -> None:
    assert candidate_harbor_versions() == tuple(
        version for version, release in CATALOG["releases"].items() if not release["selectable"]
    )
    assert "0.12" not in qualification_catalog()["releases"]


def test_harbor_020_is_selectable_with_immutable_release_evidence() -> None:
    runner = resolve_harbor_runner("0.20.0")
    release = runner.qualification_metadata()["release"]

    assert "0.20.0" in supported_harbor_versions()
    assert "0.20.0" not in candidate_harbor_versions()
    assert runner.harbor_dir == "/opt/harbor/0.20.0"
    assert release["qualification"] == "supported"
    assert release["git_tag"] == "v0.20.0"
    assert release["git_tag_object_sha"] == "f75477f2ad0b04fad199b0cb80689cc23a06c72d"
    assert release["git_commit_sha"] == "459ff6ec99417589b7f679d14ddf3b3f0ae4f1dc"
    assert release["wheel_sha256"] == ("4b7e48223aea2384cdb8c9eff35eaebd482fc9b1ec09f8193a121c47356ff19a")
    assert release["extras"] == ["opensandbox"]
    assert "opensandbox-import" in CATALOG["validation"]["checks"]


def test_unsupported_version_lists_supported_choices() -> None:
    with pytest.raises(UnsupportedHarborVersion) as exc_info:
        resolve_harbor_runner("latest")

    message = str(exc_info.value)
    for version in supported_harbor_versions():
        assert version in message
    assert "default" in message
    assert "stable" in message
