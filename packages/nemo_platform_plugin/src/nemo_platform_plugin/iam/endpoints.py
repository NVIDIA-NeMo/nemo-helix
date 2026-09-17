# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Auth service IAM API."""

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.client.endpoint import delete, get, post
from nemo_platform_plugin.client.types import BinaryContent, Paginated
from nemo_platform_plugin.iam.types import (
    AuthzRequest,
    AuthzResponse,
    ListRoleBindingsQueryParams,
    RoleBinding,
    RoleBindingDeleteResponse,
    RoleBindingInput,
    RolePropagationQueryParams,
)


@get("/apis/auth/v2/iam/role-bindings")
@abstractmethod
def list_role_bindings(*, query_params: ListRoleBindingsQueryParams | None = None) -> Paginated[RoleBinding]: ...


@post("/apis/auth/v2/iam/role-bindings")
@abstractmethod
def create_role_binding(
    *, body: RoleBindingInput, query_params: RolePropagationQueryParams | None = None
) -> RoleBinding: ...


@get("/apis/auth/v2/iam/role-bindings/{name}")
@abstractmethod
def get_role_binding(*, name: str) -> RoleBinding: ...


@delete("/apis/auth/v2/iam/role-bindings/{name}")
@abstractmethod
def revoke_role_binding(
    *, name: str, query_params: RolePropagationQueryParams | None = None
) -> RoleBindingDeleteResponse: ...


@post("/apis/auth/v2/authz/{entrypoint}")
@abstractmethod
def evaluate_authorization(*, entrypoint: str, body: AuthzRequest) -> AuthzResponse: ...


@get("/apis/auth/v2/iam/opa-bundle.tar.gz", additional_success_status_codes=(304,))
@abstractmethod
def get_opa_bundle() -> BinaryContent: ...
