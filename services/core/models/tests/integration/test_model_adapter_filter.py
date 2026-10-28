# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filtering models on whether they have adapters, against a real entity store.

``adapters`` is an entity relationship, so ``filter[adapters]=true`` must become the
relationship's ``$exists`` check; the generic null check inverted true and false.
"""

import uuid
from collections.abc import Generator

import pytest
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nhx.core.files.service import FilesService
from nhx.core.models.service import ModelsService
from nhx.testing import ClientContext, create_test_client

MODELS = "/apis/models/v2/workspaces/default/models"


@pytest.fixture(scope="module")
def clients() -> Generator[ClientContext, None, None]:
    with create_test_client(ModelsService, FilesService, client_type=ClientContext) as ctx:
        yield ctx


@pytest.fixture(scope="module")
def model_names(clients: ClientContext) -> dict[str, str]:
    """Create one model with a LoRA adapter and one without."""
    suffix = uuid.uuid4().hex[:8]
    fileset = f"fs-{suffix}"
    files = FilesClient.from_client(clients.client)
    files.create_fileset(workspace="default", body=CreateFilesetRequest(name=fileset))
    files.upload_file(workspace="default", name=fileset, path="weights.bin", content=b"x")

    names = {"with": f"with-adapter-{suffix}", "without": f"without-adapter-{suffix}"}
    for name in names.values():
        response = clients.test_client.post(MODELS, json={"name": name, "fileset": f"default/{fileset}"})
        assert response.status_code == 201, response.text
    response = clients.test_client.post(
        f"{MODELS}/{names['with']}/adapters",
        json={"name": f"adapter-{suffix}", "fileset": f"default/{fileset}", "finetuning_type": "lora"},
    )
    assert response.status_code == 201, response.text
    return names


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"filter[adapters]": "true"}, "with"),
        ({"filter[adapters]": "false"}, "without"),
        ({"filter": '{"adapters": true}'}, "with"),
        ({"filter[adapters][$exists]": "true"}, "with"),
        ({"filter[adapters][$exists]": "false"}, "without"),
    ],
)
def test_filter_models_by_adapter_presence(
    clients: ClientContext, model_names: dict[str, str], params: dict[str, str], expected: str
) -> None:
    response = clients.test_client.get(MODELS, params={**params, "page_size": 100})

    assert response.status_code == 200, response.text
    listed = {model["name"] for model in response.json()["data"]} & set(model_names.values())
    assert listed == {model_names[expected]}
