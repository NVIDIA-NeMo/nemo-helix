# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable

import httpx

from ..._types import Body, Omit, Query, Headers, NoneType, NotGiven, omit, not_given
from ..._utils import path_template, maybe_transform, async_maybe_transform
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ...pagination import SyncDefaultPagination, AsyncDefaultPagination
from ..._base_client import AsyncPaginator, make_request_options
from ...types.inference import (
    virtual_model_list_params,
    virtual_model_patch_params,
    virtual_model_create_params,
    virtual_model_delete_params,
)
from ...types.inference.virtual_model import VirtualModel
from ...types.inference.middleware_call_param import MiddlewareCallParam
from ...types.inference.virtual_model_filter_param import VirtualModelFilterParam
from ...types.inference.virtual_model_inference_config_param import VirtualModelInferenceConfigParam
from ..._exceptions import ConflictError

__all__ = ["VirtualModelsResource", "AsyncVirtualModelsResource"]


class VirtualModelsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> VirtualModelsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return VirtualModelsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> VirtualModelsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return VirtualModelsResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        autoprovisioned: bool | Omit = omit,
        default_model_entity: str | Omit = omit,
        models: Iterable[VirtualModelInferenceConfigParam] | Omit = omit,
        override_proxy: str | Omit = omit,
        post_response_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        request_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        response_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> VirtualModel:
        """
        Create a new VirtualModel in the given workspace.

        A VirtualModel defines an ordered middleware pipeline that IGW executes when an
        inference request arrives with `model: "workspace/name"` matching this entity.

        Args:
          name: Name of the virtual model within the workspace. Must be unique per workspace.

          autoprovisioned: Marks this VirtualModel as controller-managed. The Models controller will delete
              it once no ModelProvider serves the matching entity. Setting this manually opts
              the VirtualModel into that cleanup behavior.

          default_model_entity: Model entity to route to, in "workspace/name" format. Written into
              request["model"] before the request middleware pipeline runs. If omitted, a
              request middleware plugin must handle backend routing itself. Set to null to
              clear an existing value.

          models: Model entity references used by this VirtualModel. A per-entry backend_format
              overrides the referenced ModelEntity backend_format when IGW resolves the
              backend format for a request.

          override_proxy: Plugin-provided proxy implementation for IGW to use instead of its default
              aiohttp proxy. Format: "plugin-name.proxy-name". Leave unset to use the default
              IGW proxy. Set to null to clear an existing value.

          post_response_middleware: Ordered list of middleware plugins invoked after the response has been returned
              to the caller. Intended for fire-and-forget work (logging, analytics) that must
              not block or modify the response.

          request_middleware: Ordered list of middleware plugins applied before proxying to the backend. Each
              entry is a MiddlewareCall with a "name" (plugin identifier) and optional
              "config_type" and "config_id" fields that reference a stored plugin
              configuration.

          response_middleware: Ordered list of middleware plugins applied after the backend response is
              received, before returning it to the caller.


          exist_ok: Do not raise an error if the resource already exists. Returns the existing resource.


          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        try:
            if workspace is None:
                workspace = self._client._get_workspace_path_param()
            if not workspace:
                raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
            return self._post(
                path_template("/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models", workspace=workspace),
                body=maybe_transform(
                    {
                        "name": name,
                        "autoprovisioned": autoprovisioned,
                        "default_model_entity": default_model_entity,
                        "models": models,
                        "override_proxy": override_proxy,
                        "post_response_middleware": post_response_middleware,
                        "request_middleware": request_middleware,
                        "response_middleware": response_middleware,
                    },
                    virtual_model_create_params.VirtualModelCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=VirtualModel,
            )
        except ConflictError:
            if not exist_ok:
                raise
            return self.retrieve(name = name, workspace = workspace)

    def retrieve(
        self,
        name: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> VirtualModel:
        """
        Get a VirtualModel by workspace and name.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._get(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}",
                workspace=workspace,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=VirtualModel,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        exclude_autoprovisioned: bool | Omit = omit,
        filter: VirtualModelFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[VirtualModel]:
        """
        List VirtualModels for the given workspace.

        Use `workspace=-` to list across all workspaces accessible to the caller.

        Args:
          exclude_autoprovisioned: When true, controller-managed (autoprovisioned) passthrough VirtualModels are
              excluded from the results.

          filter: Filter virtual models by workspace, project, name, default_model_entity,
              guardrail_config, created_at, and updated_at.

          page: Page number (1-indexed).

          page_size: Number of results per page.

          sort: Sort field. Prefix with `-` for descending order.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return self._get_api_list(
            path_template("/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models", workspace=workspace),
            page=SyncDefaultPagination[VirtualModel],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "exclude_autoprovisioned": exclude_autoprovisioned,
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    virtual_model_list_params.VirtualModelListParams,
                ),
            ),
            model=VirtualModel,
        )

    def delete(
        self,
        name: str,
        *,
        workspace: str | None = None,
        expected_db_version: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Permanently delete a VirtualModel.

        This does not affect any in-flight requests already being routed through this
        VirtualModel. IGW's model cache is refreshed on its next polling cycle.

        Args:
          expected_db_version: Optional database version for optimistic locking. Delete only succeeds if the
              VirtualModel still has this version.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return self._delete(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}",
                workspace=workspace,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {"expected_db_version": expected_db_version}, virtual_model_delete_params.VirtualModelDeleteParams
                ),
            ),
            cast_to=NoneType,
        )

    def patch(
        self,
        name: str,
        *,
        workspace: str | None = None,
        autoprovisioned: bool | Omit = omit,
        default_model_entity: str | Omit = omit,
        models: Iterable[VirtualModelInferenceConfigParam] | Omit = omit,
        override_proxy: str | Omit = omit,
        post_response_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        request_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        response_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> VirtualModel:
        """
        Partially update a VirtualModel.

        Only fields present in the request body are modified. Fields absent from the
        request body retain their current values.

        Args:
          autoprovisioned: Marks this VirtualModel as controller-managed. The Models controller will delete
              it once no ModelProvider serves the matching entity. Setting this manually opts
              the VirtualModel into that cleanup behavior.

          default_model_entity: Model entity to route to, in "workspace/name" format. Written into
              request["model"] before the request middleware pipeline runs. If omitted, a
              request middleware plugin must handle backend routing itself. Set to null to
              clear an existing value.

          models: Model entity references used by this VirtualModel. A per-entry backend_format
              overrides the referenced ModelEntity backend_format when IGW resolves the
              backend format for a request.

          override_proxy: Plugin-provided proxy implementation for IGW to use instead of its default
              aiohttp proxy. Format: "plugin-name.proxy-name". Leave unset to use the default
              IGW proxy. Set to null to clear an existing value.

          post_response_middleware: Ordered list of middleware plugins invoked after the response has been returned
              to the caller. Intended for fire-and-forget work (logging, analytics) that must
              not block or modify the response.

          request_middleware: Ordered list of middleware plugins applied before proxying to the backend. Each
              entry is a MiddlewareCall with a "name" (plugin identifier) and optional
              "config_type" and "config_id" fields that reference a stored plugin
              configuration.

          response_middleware: Ordered list of middleware plugins applied after the backend response is
              received, before returning it to the caller.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return self._patch(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}",
                workspace=workspace,
                name=name,
            ),
            body=maybe_transform(
                {
                    "autoprovisioned": autoprovisioned,
                    "default_model_entity": default_model_entity,
                    "models": models,
                    "override_proxy": override_proxy,
                    "post_response_middleware": post_response_middleware,
                    "request_middleware": request_middleware,
                    "response_middleware": response_middleware,
                },
                virtual_model_patch_params.VirtualModelPatchParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=VirtualModel,
        )


class AsyncVirtualModelsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncVirtualModelsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncVirtualModelsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncVirtualModelsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncVirtualModelsResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        name: str,
        autoprovisioned: bool | Omit = omit,
        default_model_entity: str | Omit = omit,
        models: Iterable[VirtualModelInferenceConfigParam] | Omit = omit,
        override_proxy: str | Omit = omit,
        post_response_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        request_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        response_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> VirtualModel:
        """
        Create a new VirtualModel in the given workspace.

        A VirtualModel defines an ordered middleware pipeline that IGW executes when an
        inference request arrives with `model: "workspace/name"` matching this entity.

        Args:
          name: Name of the virtual model within the workspace. Must be unique per workspace.

          autoprovisioned: Marks this VirtualModel as controller-managed. The Models controller will delete
              it once no ModelProvider serves the matching entity. Setting this manually opts
              the VirtualModel into that cleanup behavior.

          default_model_entity: Model entity to route to, in "workspace/name" format. Written into
              request["model"] before the request middleware pipeline runs. If omitted, a
              request middleware plugin must handle backend routing itself. Set to null to
              clear an existing value.

          models: Model entity references used by this VirtualModel. A per-entry backend_format
              overrides the referenced ModelEntity backend_format when IGW resolves the
              backend format for a request.

          override_proxy: Plugin-provided proxy implementation for IGW to use instead of its default
              aiohttp proxy. Format: "plugin-name.proxy-name". Leave unset to use the default
              IGW proxy. Set to null to clear an existing value.

          post_response_middleware: Ordered list of middleware plugins invoked after the response has been returned
              to the caller. Intended for fire-and-forget work (logging, analytics) that must
              not block or modify the response.

          request_middleware: Ordered list of middleware plugins applied before proxying to the backend. Each
              entry is a MiddlewareCall with a "name" (plugin identifier) and optional
              "config_type" and "config_id" fields that reference a stored plugin
              configuration.

          response_middleware: Ordered list of middleware plugins applied after the backend response is
              received, before returning it to the caller.


          exist_ok: Do not raise an error if the resource already exists. Returns the existing resource.


          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        try:
            if workspace is None:
                workspace = self._client._get_workspace_path_param()
            if not workspace:
                raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
            return await self._post(
                path_template("/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models", workspace=workspace),
                body=await async_maybe_transform(
                    {
                        "name": name,
                        "autoprovisioned": autoprovisioned,
                        "default_model_entity": default_model_entity,
                        "models": models,
                        "override_proxy": override_proxy,
                        "post_response_middleware": post_response_middleware,
                        "request_middleware": request_middleware,
                        "response_middleware": response_middleware,
                    },
                    virtual_model_create_params.VirtualModelCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=VirtualModel,
            )
        except ConflictError:
            if not exist_ok:
                raise
            return await self.retrieve(name = name, workspace = workspace)

    async def retrieve(
        self,
        name: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> VirtualModel:
        """
        Get a VirtualModel by workspace and name.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return await self._get(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}",
                workspace=workspace,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=VirtualModel,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        exclude_autoprovisioned: bool | Omit = omit,
        filter: VirtualModelFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[VirtualModel, AsyncDefaultPagination[VirtualModel]]:
        """
        List VirtualModels for the given workspace.

        Use `workspace=-` to list across all workspaces accessible to the caller.

        Args:
          exclude_autoprovisioned: When true, controller-managed (autoprovisioned) passthrough VirtualModels are
              excluded from the results.

          filter: Filter virtual models by workspace, project, name, default_model_entity,
              guardrail_config, created_at, and updated_at.

          page: Page number (1-indexed).

          page_size: Number of results per page.

          sort: Sort field. Prefix with `-` for descending order.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        return self._get_api_list(
            path_template("/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models", workspace=workspace),
            page=AsyncDefaultPagination[VirtualModel],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "exclude_autoprovisioned": exclude_autoprovisioned,
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    virtual_model_list_params.VirtualModelListParams,
                ),
            ),
            model=VirtualModel,
        )

    async def delete(
        self,
        name: str,
        *,
        workspace: str | None = None,
        expected_db_version: int | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Permanently delete a VirtualModel.

        This does not affect any in-flight requests already being routed through this
        VirtualModel. IGW's model cache is refreshed on its next polling cycle.

        Args:
          expected_db_version: Optional database version for optimistic locking. Delete only succeeds if the
              VirtualModel still has this version.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return await self._delete(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}",
                workspace=workspace,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {"expected_db_version": expected_db_version}, virtual_model_delete_params.VirtualModelDeleteParams
                ),
            ),
            cast_to=NoneType,
        )

    async def patch(
        self,
        name: str,
        *,
        workspace: str | None = None,
        autoprovisioned: bool | Omit = omit,
        default_model_entity: str | Omit = omit,
        models: Iterable[VirtualModelInferenceConfigParam] | Omit = omit,
        override_proxy: str | Omit = omit,
        post_response_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        request_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        response_middleware: Iterable[MiddlewareCallParam] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> VirtualModel:
        """
        Partially update a VirtualModel.

        Only fields present in the request body are modified. Fields absent from the
        request body retain their current values.

        Args:
          autoprovisioned: Marks this VirtualModel as controller-managed. The Models controller will delete
              it once no ModelProvider serves the matching entity. Setting this manually opts
              the VirtualModel into that cleanup behavior.

          default_model_entity: Model entity to route to, in "workspace/name" format. Written into
              request["model"] before the request middleware pipeline runs. If omitted, a
              request middleware plugin must handle backend routing itself. Set to null to
              clear an existing value.

          models: Model entity references used by this VirtualModel. A per-entry backend_format
              overrides the referenced ModelEntity backend_format when IGW resolves the
              backend format for a request.

          override_proxy: Plugin-provided proxy implementation for IGW to use instead of its default
              aiohttp proxy. Format: "plugin-name.proxy-name". Leave unset to use the default
              IGW proxy. Set to null to clear an existing value.

          post_response_middleware: Ordered list of middleware plugins invoked after the response has been returned
              to the caller. Intended for fire-and-forget work (logging, analytics) that must
              not block or modify the response.

          request_middleware: Ordered list of middleware plugins applied before proxying to the backend. Each
              entry is a MiddlewareCall with a "name" (plugin identifier) and optional
              "config_type" and "config_id" fields that reference a stored plugin
              configuration.

          response_middleware: Ordered list of middleware plugins applied after the backend response is
              received, before returning it to the caller.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        if not workspace:
            raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
        if not name:
            raise ValueError(f"Expected a non-empty value for `name` but received {name!r}")
        return await self._patch(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}",
                workspace=workspace,
                name=name,
            ),
            body=await async_maybe_transform(
                {
                    "autoprovisioned": autoprovisioned,
                    "default_model_entity": default_model_entity,
                    "models": models,
                    "override_proxy": override_proxy,
                    "post_response_middleware": post_response_middleware,
                    "request_middleware": request_middleware,
                    "response_middleware": response_middleware,
                },
                virtual_model_patch_params.VirtualModelPatchParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=VirtualModel,
        )


class VirtualModelsResourceWithRawResponse:
    def __init__(self, virtual_models: VirtualModelsResource) -> None:
        self._virtual_models = virtual_models

        self.create = to_raw_response_wrapper(
            virtual_models.create,
        )
        self.retrieve = to_raw_response_wrapper(
            virtual_models.retrieve,
        )
        self.list = to_raw_response_wrapper(
            virtual_models.list,
        )
        self.delete = to_raw_response_wrapper(
            virtual_models.delete,
        )
        self.patch = to_raw_response_wrapper(
            virtual_models.patch,
        )


class AsyncVirtualModelsResourceWithRawResponse:
    def __init__(self, virtual_models: AsyncVirtualModelsResource) -> None:
        self._virtual_models = virtual_models

        self.create = async_to_raw_response_wrapper(
            virtual_models.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            virtual_models.retrieve,
        )
        self.list = async_to_raw_response_wrapper(
            virtual_models.list,
        )
        self.delete = async_to_raw_response_wrapper(
            virtual_models.delete,
        )
        self.patch = async_to_raw_response_wrapper(
            virtual_models.patch,
        )


class VirtualModelsResourceWithStreamingResponse:
    def __init__(self, virtual_models: VirtualModelsResource) -> None:
        self._virtual_models = virtual_models

        self.create = to_streamed_response_wrapper(
            virtual_models.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            virtual_models.retrieve,
        )
        self.list = to_streamed_response_wrapper(
            virtual_models.list,
        )
        self.delete = to_streamed_response_wrapper(
            virtual_models.delete,
        )
        self.patch = to_streamed_response_wrapper(
            virtual_models.patch,
        )


class AsyncVirtualModelsResourceWithStreamingResponse:
    def __init__(self, virtual_models: AsyncVirtualModelsResource) -> None:
        self._virtual_models = virtual_models

        self.create = async_to_streamed_response_wrapper(
            virtual_models.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            virtual_models.retrieve,
        )
        self.list = async_to_streamed_response_wrapper(
            virtual_models.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            virtual_models.delete,
        )
        self.patch = async_to_streamed_response_wrapper(
            virtual_models.patch,
        )
