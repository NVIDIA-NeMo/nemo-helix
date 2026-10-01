# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid

from nhx.testing import ClientContext

DEFAULT_WORKSPACE = "default"
MODELS_PATH = f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models"


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
    name: str,
    family: str | None = None,
    model_providers: list[str] | None = None,
) -> None:
    body: dict = {"name": name}
    if family is not None:
        body["spec"] = _spec_with_family(family)
    if model_providers is not None:
        body["model_providers"] = model_providers
    response = test_clients.test_client.post(MODELS_PATH, json=body)
    assert response.status_code == 201, f"Failed to create model: {response.text}"


def _list_names(test_clients: ClientContext, params: dict[str, str]) -> list[str]:
    response = test_clients.test_client.get(MODELS_PATH, params=params)
    assert response.status_code == 200, response.text
    return sorted(m["name"] for m in response.json()["data"])


def test_family_filter_returns_only_matching_family(test_clients: ClientContext):
    family = _uid("fam")
    llama_a = _uid("llama-a")
    llama_b = _uid("llama-b")
    other = _uid("other")
    _create_model(test_clients, llama_a, family=family)
    _create_model(test_clients, llama_b, family=family)
    _create_model(test_clients, other, family=_uid("fam"))

    assert _list_names(test_clients, {"filter[family]": family}) == sorted([llama_a, llama_b])


def test_family_filter_combined_with_name_filter(test_clients: ClientContext):
    family = _uid("fam")
    first = _uid("first")
    second = _uid("second")
    _create_model(test_clients, first, family=family)
    _create_model(test_clients, second, family=family)

    assert _list_names(test_clients, {"filter[family]": family, "filter[name]": first}) == [first]


def test_model_providers_filter_matches_models_linked_to_provider(test_clients: ClientContext):
    provider = f"{DEFAULT_WORKSPACE}/{_uid('provider')}"
    linked = _uid("linked")
    unlinked = _uid("unlinked")
    _create_model(test_clients, linked, model_providers=[provider, f"{DEFAULT_WORKSPACE}/{_uid('extra')}"])
    _create_model(test_clients, unlinked, model_providers=[f"{DEFAULT_WORKSPACE}/{_uid('other')}"])

    assert _list_names(test_clients, {"filter[model_providers]": provider}) == [linked]


def test_model_providers_filter_does_not_match_provider_name_prefix(test_clients: ClientContext):
    base = _uid("provider")
    provider = f"{DEFAULT_WORKSPACE}/{base}"
    longer_provider = f"{DEFAULT_WORKSPACE}/{base}-v2"
    exact = _uid("exact")
    prefixed = _uid("prefixed")
    _create_model(test_clients, exact, model_providers=[provider])
    _create_model(test_clients, prefixed, model_providers=[longer_provider])

    assert _list_names(test_clients, {"filter[model_providers]": provider}) == [exact]


def test_model_providers_filter_nested_in_or_is_rejected(test_clients: ClientContext):
    provider = f"{DEFAULT_WORKSPACE}/{_uid('provider')}"
    nested = json.dumps({"$or": [{"model_providers": provider}, {"name": _uid("any")}]})

    response = test_clients.test_client.get(MODELS_PATH, params={"filter": nested})

    assert response.status_code == 400, response.text
    assert "model_providers" in response.text


def test_family_and_model_providers_filters_combine(test_clients: ClientContext):
    family = _uid("fam")
    provider = f"{DEFAULT_WORKSPACE}/{_uid('provider')}"
    both = _uid("both")
    family_only = _uid("family-only")
    provider_only = _uid("provider-only")
    _create_model(test_clients, both, family=family, model_providers=[provider])
    _create_model(test_clients, family_only, family=family)
    _create_model(test_clients, provider_only, family=_uid("fam"), model_providers=[provider])

    params = {"filter[family]": family, "filter[model_providers]": provider}
    assert _list_names(test_clients, params) == [both]
