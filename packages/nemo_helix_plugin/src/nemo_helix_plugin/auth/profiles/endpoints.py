# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.auth.profiles.types import (
    Me,
    PreferenceDefaults,
    PreferenceDefaultsList,
    PreferenceDocument,
    PreferenceDocumentList,
    PreferenceListQueryParams,
    PreferenceQueryParams,
    PreferenceValue,
    ProfileResolveRequest,
    ProfileResolveResponse,
    ProfileUpdate,
)
from nemo_helix_plugin.client.endpoint import delete, get, patch, post, put


@get("/apis/auth/v2/me")
@abstractmethod
def get_me() -> Me: ...


@patch("/apis/auth/v2/me/profile")
@abstractmethod
def update_my_profile(*, body: ProfileUpdate) -> Me: ...


@get("/apis/auth/v2/me/preferences")
@abstractmethod
def list_my_preferences(*, query_params: PreferenceListQueryParams | None = None) -> PreferenceDocumentList: ...


@get("/apis/auth/v2/me/preferences/{namespace}")
@abstractmethod
def get_my_preference(*, namespace: str, query_params: PreferenceQueryParams | None = None) -> PreferenceDocument: ...


@put("/apis/auth/v2/me/preferences/{namespace}")
@abstractmethod
def replace_my_preference(
    *, namespace: str, body: PreferenceValue, query_params: PreferenceQueryParams | None = None
) -> PreferenceDocument: ...


@patch("/apis/auth/v2/me/preferences/{namespace}")
@abstractmethod
def patch_my_preference(
    *, namespace: str, body: PreferenceValue, query_params: PreferenceQueryParams | None = None
) -> PreferenceDocument: ...


@delete("/apis/auth/v2/me/preferences/{namespace}")
@abstractmethod
def delete_my_preference(*, namespace: str, query_params: PreferenceQueryParams | None = None) -> None: ...


@get("/apis/auth/v2/preferences/defaults")
@abstractmethod
def list_preference_defaults() -> PreferenceDefaultsList: ...


@get("/apis/auth/v2/preferences/defaults/{namespace}")
@abstractmethod
def get_preference_defaults(*, namespace: str) -> PreferenceDefaults: ...


@put("/apis/auth/v2/preferences/defaults/{namespace}")
@abstractmethod
def replace_preference_defaults(*, namespace: str, body: PreferenceValue) -> PreferenceDefaults: ...


@patch("/apis/auth/v2/preferences/defaults/{namespace}")
@abstractmethod
def patch_preference_defaults(*, namespace: str, body: PreferenceValue) -> PreferenceDefaults: ...


@delete("/apis/auth/v2/preferences/defaults/{namespace}")
@abstractmethod
def delete_preference_defaults(*, namespace: str) -> None: ...


@post("/apis/auth/v2/profiles/resolve")
@abstractmethod
def resolve_profiles(*, body: ProfileResolveRequest) -> ProfileResolveResponse: ...
