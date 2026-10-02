# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid

from nhx.testing import ClientContext

DEFAULT_WORKSPACE = "default"
MODELS_PATH = f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models"
FAMILIES_PATH = f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/model-families"


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _spec_with_family(family: str) -> dict:
    return {
        "context_size": 4096,
        "is_chat": True,
        "checkpoint_model_name": "meta-llama/Llama-3.2-1b-instruct",
        "family": family,
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 32,
        "ffn_hidden_size": 16384,
        "vocab_size": 32000,
        "tied_embeddings": True,
        "gated_mlp": True,
        "base_num_parameters": 7000000000,
        "precision": "fp16",
    }


def _create_model(
    test_clients: ClientContext,
    family: str | None = None,
    model_providers: list[str] | None = None,
) -> None:
    body: dict = {"name": _uid("model")}
    if family is not None:
        body["spec"] = _spec_with_family(family)
    if model_providers is not None:
        body["model_providers"] = model_providers
    response = test_clients.test_client.post(MODELS_PATH, json=body)
    assert response.status_code == 201, f"Failed to create model: {response.text}"


def _list_families(test_clients: ClientContext, params: dict[str, str] | None = None) -> dict:
    response = test_clients.test_client.get(FAMILIES_PATH, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _counts(body: dict) -> dict[str, int]:
    return {family["name"]: family["model_count"] for family in body["data"]}


def test_lists_each_family_once_with_its_model_count(test_clients: ClientContext):
    llama = _uid("llama")
    mixtral = _uid("mixtral")
    _create_model(test_clients, family=llama)
    _create_model(test_clients, family=llama)
    _create_model(test_clients, family=mixtral)

    counts = _counts(_list_families(test_clients))

    assert counts[llama] == 2
    assert counts[mixtral] == 1


def test_leaves_out_models_without_a_family(test_clients: ClientContext):
    family = _uid("fam")
    _create_model(test_clients, family=family)
    _create_model(test_clients)

    body = _list_families(test_clients)

    assert family in _counts(body)
    assert all(entry["name"] for entry in body["data"])


def test_sorts_by_name_in_either_direction(test_clients: ClientContext):
    prefix = _uid("sorted")
    families = [f"{prefix}-b", f"{prefix}-a", f"{prefix}-c"]
    for family in families:
        _create_model(test_clients, family=family)

    ascending = [f["name"] for f in _list_families(test_clients, {"sort": "name"})["data"]]
    descending = [f["name"] for f in _list_families(test_clients, {"sort": "-name"})["data"]]

    mine = [name for name in ascending if name.startswith(prefix)]
    assert mine == sorted(families)
    assert [name for name in descending if name.startswith(prefix)] == sorted(families, reverse=True)


def test_sorts_by_model_count_descending(test_clients: ClientContext):
    prefix = _uid("counted")
    many, few = f"{prefix}-many", f"{prefix}-few"
    for _ in range(3):
        _create_model(test_clients, family=many)
    _create_model(test_clients, family=few)

    body = _list_families(test_clients, {"sort": "-model_count"})

    names = [entry["name"] for entry in body["data"] if entry["name"].startswith(prefix)]
    assert names == [many, few]


def test_paginates_the_families(test_clients: ClientContext):
    prefix = _uid("paged")
    for suffix in ("a", "b", "c"):
        _create_model(test_clients, family=f"{prefix}-{suffix}")

    body = _list_families(test_clients, {"page_size": "1", "page": "2", "sort": "name"})

    assert body["pagination"]["page"] == 2
    assert body["pagination"]["page_size"] == 1
    assert body["pagination"]["current_page_size"] == 1
    assert body["pagination"]["total_results"] >= 3
    assert len(body["data"]) == 1


def test_narrows_the_families_by_provider(test_clients: ClientContext):
    provider = f"{DEFAULT_WORKSPACE}/{_uid('provider')}"
    served, unserved = _uid("served"), _uid("unserved")
    _create_model(test_clients, family=served, model_providers=[provider])
    _create_model(test_clients, family=unserved)

    counts = _counts(_list_families(test_clients, {"filter[model_providers]": provider}))

    assert counts == {served: 1}


def test_rejects_a_nested_provider_filter(test_clients: ClientContext):
    provider = f"{DEFAULT_WORKSPACE}/{_uid('provider')}"
    nested = json.dumps({"$or": [{"model_providers": provider}, {"name": _uid("any")}]})

    response = test_clients.test_client.get(FAMILIES_PATH, params={"filter": nested})

    assert response.status_code == 400, response.text


def test_rejects_out_of_range_pagination(test_clients: ClientContext):
    for params in ({"page_size": "0"}, {"page_size": "1001"}, {"page": "0"}):
        response = test_clients.test_client.get(FAMILIES_PATH, params=params)

        assert response.status_code == 422, (params, response.text)


def test_passes_through_an_entity_store_client_error(test_clients: ClientContext):
    response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{_uid('missing')}/model-families")

    assert 400 <= response.status_code < 500, response.text
