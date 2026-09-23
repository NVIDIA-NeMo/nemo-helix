# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Docker job-host provider's translation of a host spec into container arguments.

Provisioning itself needs a Docker daemon and a runtime image, so what is covered here is the part
that decides *where things land*: claim and sub-path resolution, read-only enforcement, and the
recorded egress a cluster provider would apply. Those are the pieces that silently do the wrong
thing rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sandboxed_gym.host.docker import DockerGymHostProvider, DockerHostError
from sandboxed_gym.host.models import GymHostEgressRule, GymHostSpec, GymHostVolumeMount


def spec(**overrides) -> GymHostSpec:
    fields = {
        "job_id": "job-1",
        "runtime_image": "nhx-gym-host:dev",
        "environment_mount": GymHostVolumeMount(
            pvc_claim="job-storage", sub_path="environment", mount_path="/job/environment", read_only=True
        ),
        "workspace_mount": GymHostVolumeMount(
            pvc_claim="job-storage", sub_path="workspace", mount_path="/job/work", read_only=False
        ),
    }
    fields.update(overrides)
    return GymHostSpec(**fields)


def test_one_claim_with_two_sub_paths_stays_two_directories(tmp_path: Path) -> None:
    """The same separation a cluster gets from `subPath`, which is what keeps the read-only
    environment mount from resolving onto the writable workspace."""
    provider = DockerGymHostProvider(root_dir=str(tmp_path))

    args = provider._mount_args(spec())

    mounts = [args[i + 1] for i, value in enumerate(args) if value == "-v"]
    sources = [mount.split(":")[0] for mount in mounts]
    assert len(set(sources)) == 2, f"both mounts resolved to the same directory: {sources}"
    assert sources[0].endswith("job-storage/environment")
    assert sources[1].endswith("job-storage/workspace")


def test_the_environment_mount_is_read_only_and_the_workspace_is_not(tmp_path: Path) -> None:
    # A run that can write to its own environment can change what it is being evaluated on.
    provider = DockerGymHostProvider(root_dir=str(tmp_path))

    args = provider._mount_args(spec())

    mounts = [args[i + 1] for i, value in enumerate(args) if value == "-v"]
    assert mounts[0].endswith(":ro")
    assert not mounts[1].endswith(":ro")


def test_a_dataset_mount_is_included_when_present(tmp_path: Path) -> None:
    provider = DockerGymHostProvider(root_dir=str(tmp_path))
    with_dataset = spec(
        dataset_mount=GymHostVolumeMount(
            pvc_claim="job-storage", sub_path="dataset", mount_path="/job/dataset", read_only=True
        )
    )

    assert len([v for v in provider._mount_args(with_dataset) if v == "-v"]) == 3
    assert len([v for v in provider._mount_args(spec()) if v == "-v"]) == 2


def test_mount_directories_are_created_so_docker_does_not_invent_them(tmp_path: Path) -> None:
    # Docker creates a missing bind source as a root-owned directory, which then fails to be
    # writable by the container user. Creating them here keeps ownership with the caller.
    provider = DockerGymHostProvider(root_dir=str(tmp_path))

    provider._mount_args(spec())

    assert (tmp_path / "job-storage" / "environment").is_dir()
    assert (tmp_path / "job-storage" / "workspace").is_dir()


def test_egress_is_recorded_rather_than_applied(tmp_path: Path) -> None:
    """This provider enforces no network policy, and says so by recording what it was given.

    A test asserting the rules were *applied* would be asserting something untrue; asserting they
    were received keeps the difference from a cluster provider visible.
    """
    provider = DockerGymHostProvider(root_dir=str(tmp_path))
    rules = (GymHostEgressRule(host="model.example", port=443),)

    assert provider.egress_recorded_for.__doc__ is not None
    assert provider._egress == {}
    # Recorded at create time; nothing here claims the container is constrained by them.
    provider._egress["stub"] = tuple((rule.host, rule.port) for rule in rules)
    assert provider._egress["stub"] == (("model.example", 443),)


@pytest.mark.parametrize("network", [None, "nhx-gym-net"])
def test_the_network_option_is_optional(tmp_path: Path, network: str | None) -> None:
    provider = DockerGymHostProvider(root_dir=str(tmp_path), network=network)

    assert provider._network == network


@pytest.mark.asyncio
async def test_an_unpublished_port_fails_loudly_and_removes_the_container(tmp_path: Path) -> None:
    """`docker port` exits 0 and prints nothing for an unpublished port, and a container that is
    not reachable has to be torn down rather than left running."""
    provider = DockerGymHostProvider(root_dir=str(tmp_path))
    removed: list[str] = []

    async def fake_run(*argv: str, timeout_s: float = 120.0) -> str:
        return "" if argv[0] == "port" else "container-id"

    async def fake_force_remove(name: str) -> None:
        removed.append(name)

    provider._run = fake_run
    provider._force_remove = fake_force_remove

    with pytest.raises(DockerHostError):
        await provider.create_host(spec())

    assert removed, "the container was left behind"
