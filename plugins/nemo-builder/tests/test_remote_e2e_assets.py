# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the remote backend's Kind e2e installs, checked without a cluster.

``e2e/test_builder_remote.py`` runs only on a cluster, in the ``kind-builder-e2e`` job. A misspelt setting in its
values file would otherwise show only there, as a build that never becomes ready. The stand-in step program it
mounts is run through the driver in ``test_remote_driver.py``.

Offline: nothing here renders the chart or contacts a deployment.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml
from nemo_builder_plugin import remote
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.remote import CANCEL_MARGIN_SECONDS, STEP_NAME, RemoteBackend
from nemo_builder_plugin.schema import MAX_BUILD_SPECS, BuildSet, BuildSpec, FileSetSource
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfile
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider

ROOT = Path(__file__).resolve().parents[3]
VALUES = ROOT / "e2e/k8s/values/builder-remote.yaml"
STEP_PROGRAM = ROOT / "e2e/resources/builder_remote/remote_build_step.py"
E2E_MODULE = ROOT / "e2e/test_builder_remote.py"

_TEMPLATE = re.compile(r"\{\{.*?\}\}")


def _rendered(value: Any) -> Any:
    """``value`` with a word in place of each chart template in it: enough to validate what is not templated."""
    if isinstance(value, dict):
        return {key: _rendered(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rendered(item) for item in value]
    return _TEMPLATE.sub("rendered", value) if isinstance(value, str) else value


@pytest.fixture(scope="module")
def platform_config() -> dict[str, Any]:
    """The values file's ``platformConfig``, as written."""
    return yaml.safe_load(VALUES.read_text(encoding="utf-8"))["platformConfig"]


@pytest.fixture(scope="module")
def builder_config(platform_config: dict[str, Any]) -> BuilderConfig:
    return BuilderConfig.model_validate(_rendered(platform_config["builder"]))


@pytest.fixture(scope="module")
def profile(platform_config: dict[str, Any]) -> KubernetesJobExecutionProfile:
    """The one Jobs execution profile the values file adds."""
    (executor,) = platform_config["jobs"]["executors"]
    return KubernetesJobExecutionProfile.model_validate(_rendered(executor))


def test_the_values_file_is_a_remote_backend_the_builder_accepts(
    platform_config: dict[str, Any],
    builder_config: BuilderConfig,
    profile: KubernetesJobExecutionProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A setting the builder doesn't know, or a deployment it refuses, would refuse every submit on the cluster."""
    written = platform_config["builder"]["remote"]
    # The setting takes a bare host, unlike the in-cluster `builder.registry`.
    assert "://" not in written["registry"]
    assert builder_config.remote.registry == "rendered.rendered.svc.cluster.local:5000"
    # The e2e runs it with platform auth off. Nothing here accepts a completion no one can verify.
    assert "allow_unverified_completion" not in written
    assert builder_config.remote.credential_secret is None
    monkeypatch.setattr(remote, "platform_auth_enabled", lambda: False)

    backend = load_backend(builder_config, [profile])
    assert isinstance(backend, RemoteBackend)
    # The request the e2e test submits: one unnamed spec, from a fileset, with no output.
    build_set = BuildSet(name="remote-ready", revision=1, build_specs=[BuildSpec(source=FileSetSource(fileset="ctx"))])
    plan = BuildPlan.resolve(build_set.with_default_platform(builder_config.default_platform), workspace="default")
    backend.check(plan)

    (step,) = backend.compile(plan.with_destinations(backend.destination)).steps
    assert step.name == STEP_NAME
    executor = step.executor
    assert isinstance(executor, CPUExecutionProvider)
    assert executor.profile == profile.profile
    # The release's own step image, with the stand-in's program for its command.
    assert executor.container.image == "rendered/nhx-builder-tasks:rendered"
    assert executor.container.command == ["python", f"/opt/nhx-e2e-remote-step/{STEP_PROGRAM.name}"]


def test_the_values_file_declares_the_profile_the_backend_checks_for(
    platform_config: dict[str, Any], builder_config: BuilderConfig, profile: KubernetesJobExecutionProfile
) -> None:
    settings = builder_config.remote
    assert (profile.provider, profile.profile, profile.backend) == ("cpu", settings.profile, "kubernetes_job")
    # The rule in `RemoteBackend`: a profile that ends a step before its builds could finish is refused.
    needed = settings.build_timeout_seconds * min(settings.max_specs_per_set, MAX_BUILD_SPECS) + CANCEL_MARGIN_SECONDS
    assert profile.config.ttl_seconds_active >= needed
    # The step runs in the release namespace, where the e2e test creates the ConfigMap.
    assert profile.config.namespace is None

    storage = profile.config.storage
    (volume,) = storage.additional_volumes
    (mount,) = storage.additional_volume_mounts
    assert volume.config_map is not None and mount.name == volume.name and mount.read_only
    # The command runs the file that ConfigMap holds, where it is mounted: the stand-in under e2e/resources.
    program = PurePosixPath(settings.command[-1])
    assert program.parent == PurePosixPath(mount.mount_path)
    assert program.name == STEP_PROGRAM.name and STEP_PROGRAM.is_file()
    # And the test that creates the ConfigMap names it as the profile mounts it.
    assert f'"{volume.config_map.name}"' in E2E_MODULE.read_text(encoding="utf-8")

    # Jobs' profile models ignore a key they don't know, so a misspelt one leaves its setting at the default: for
    # `launcher_image`, an image Kind can't pull. Every key the file writes is one its model took.
    (written,) = platform_config["jobs"]["executors"]
    assert profile.model_fields_set == set(written)
    assert profile.config.model_fields_set == set(written["config"])
    written_storage = written["config"]["storage"]
    assert storage.model_fields_set == set(written_storage)
    assert volume.model_fields_set == set(written_storage["additional_volumes"][0])
    assert volume.config_map.model_fields_set == set(written_storage["additional_volumes"][0]["config_map"])
    assert mount.model_fields_set == set(written_storage["additional_volume_mounts"][0])
