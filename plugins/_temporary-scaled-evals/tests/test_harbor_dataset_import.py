# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import tarfile
from unittest.mock import MagicMock

import pytest
from httpx import Request, Response

try:
    from scaled_evals.api.build.task_image_identity import (
        TaskImageIdentityError,
        normalize_upstream_image_ref,
        resolve_upstream_image,
    )
    from scaled_evals.api.settings import settings
    from scaled_evals.dispatch.harbor_dataset_images import dataset_configs, effective_image_mode
    from scaled_evals.harbor_dataset_import import (
        HarborDatasetImageImport,
        build_image_import_context,
    )
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_import_context_is_pinned_and_deterministic() -> None:
    imported = HarborDatasetImageImport.parse("docker.io/example/task@sha256:" + "a" * 64)

    first = build_image_import_context(imported)
    assert first == build_image_import_context(imported)
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        dockerfile = archive.extractfile("Dockerfile")
        assert dockerfile is not None
        assert dockerfile.read() == f"FROM {imported.source_image}\n".encode()


@pytest.mark.parametrize("image", ["docker.io/example/task:latest", "docker.io/example/task@sha256:bad"])
def test_import_rejects_unpinned_images(image: str) -> None:
    with pytest.raises(ValueError, match="pinned"):
        HarborDatasetImageImport.parse(image)


def test_import_context_carries_runnable_harbor_task(tmp_path) -> None:  # noqa: ANN001
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('[environment]\ndocker_image = "example/task:1"\n')
    (task / "instruction.md").write_text("do the thing\n")
    imported = HarborDatasetImageImport.parse("docker.io/example/task@sha256:" + "a" * 64)

    context = build_image_import_context(imported, task_dir=task, task_name="example-task")

    with tarfile.open(fileobj=io.BytesIO(context), mode="r:gz") as archive:
        assert archive.extractfile("tasks/example-task/task.toml") is not None
        instruction = archive.extractfile("tasks/example-task/instruction.md")
        assert instruction is not None
        assert instruction.read() == b"do the thing\n"


def test_upstream_image_resolution_expands_docker_shorthand() -> None:
    assert normalize_upstream_image_ref("alexgshaw/task:20251031") == ("docker.io/alexgshaw/task:20251031")
    response = Response(
        200,
        headers={"Docker-Content-Digest": "sha256:" + "a" * 64},
        request=Request("GET", "https://docker.io/v2/alexgshaw/task/manifests/20251031"),
    )
    requested: list[str] = []

    def get(_self, url, **_kwargs):  # type: ignore[no-untyped-def]
        requested.append(url)
        return response

    client = type("Client", (), {"get": get})()

    resolved = resolve_upstream_image("alexgshaw/task:20251031", client=client)

    assert resolved.immutable_ref == "docker.io/alexgshaw/task@sha256:" + "a" * 64
    assert requested == ["https://registry-1.docker.io/v2/alexgshaw/task/manifests/20251031"]


def test_upstream_resolution_enforces_registry_and_bearer_host_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    with pytest.raises(TaskImageIdentityError, match="registry.example.*not approved"):
        resolve_upstream_image("registry.example/team/task:latest", client=client)
    client.get.assert_not_called()

    monkeypatch.setattr(
        settings,
        "harbor_dataset_upstream_allowed_registries",
        "docker.io,ghcr.io,registry.example",
    )
    monkeypatch.setattr(settings, "task_image_registry_insecure", True)
    client.get.return_value = Response(
        401,
        headers={
            "WWW-Authenticate": (
                'Bearer realm="https://tokens.example/token",'
                'service="registry.example",scope="repository:team/task:pull"'
            )
        },
        request=Request("GET", "https://registry.example/v2/team/task/manifests/latest"),
    )
    with pytest.raises(TaskImageIdentityError, match="not publicly readable"):
        resolve_upstream_image("registry.example/team/task:latest", client=client)
    assert client.get.call_count == 1

    responses = iter(
        [
            Response(
                401,
                headers={
                    "WWW-Authenticate": (
                        'Bearer realm="https://auth.docker.io/token",'
                        'service="registry.docker.io",scope="repository:library/ubuntu:pull"'
                    )
                },
                request=Request("GET", "https://registry-1.docker.io/v2/library/ubuntu/manifests/latest"),
            ),
            Response(
                200,
                json={"token": "public-token"},
                request=Request("GET", "https://auth.docker.io/token"),
            ),
            Response(
                200,
                headers={"Docker-Content-Digest": "sha256:" + "a" * 64},
                request=Request("GET", "https://registry-1.docker.io/v2/library/ubuntu/manifests/latest"),
            ),
        ]
    )
    docker_client = MagicMock()
    docker_client.get.side_effect = lambda *_args, **_kwargs: next(responses)
    assert resolve_upstream_image("ubuntu:latest", client=docker_client).digest == "sha256:" + "a" * 64
    assert docker_client.get.call_count == 3
    assert docker_client.get.call_args_list[0].args[0].startswith("https://registry-1.docker.io/")


def test_dataset_image_mode_is_always_managed() -> None:
    assert effective_image_mode({"dataset_only": True}) == "managed"
    assert effective_image_mode({"dataset_only": True, "dataset_image_mode": "managed"}) == ("managed")
    with pytest.raises(ValueError, match="deployment-managed"):
        effective_image_mode({"dataset_only": True, "dataset_image_mode": "direct"})


def test_dataset_configs_reads_wrapped_harbor_yaml() -> None:
    profile = {
        "dataset_only": True,
        "harbor_config": "datasets:\n  - name: org/dataset\n    n_tasks: 10\n",
    }

    assert dataset_configs(profile) == [{"name": "org/dataset", "n_tasks": 10}]
