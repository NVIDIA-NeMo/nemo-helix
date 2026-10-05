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

from typing import Dict

import httpx

from ...._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ...._utils import path_template, maybe_transform, async_maybe_transform
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ...._base_client import make_request_options
from ....types.inference.gateway import model_put_params, model_post_params, model_patch_params
from ....types.inference.gateway.model_put_response import ModelPutResponse
from ....types.inference.gateway.model_post_response import ModelPostResponse
from ....types.inference.gateway.model_patch_response import ModelPatchResponse

__all__ = ["ModelResource", "AsyncModelResource"]


class ModelResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> ModelResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return ModelResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ModelResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return ModelResourceWithStreamingResponse(self)

    def delete(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> object:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._delete(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )

    def get(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> object:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._get(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )

    def patch(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelPatchResponse:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._patch(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=maybe_transform(body, model_patch_params.ModelPatchParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelPatchResponse,
        )

    def post(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelPostResponse:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._post(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=maybe_transform(body, model_post_params.ModelPostParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelPostResponse,
        )

    def put(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelPutResponse:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._put(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=maybe_transform(body, model_put_params.ModelPutParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelPutResponse,
        )


class AsyncModelResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncModelResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncModelResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncModelResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncModelResourceWithStreamingResponse(self)

    async def delete(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> object:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._delete(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )

    async def get(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> object:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._get(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=object,
        )

    async def patch(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelPatchResponse:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._patch(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=await async_maybe_transform(body, model_patch_params.ModelPatchParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelPatchResponse,
        )

    async def post(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelPostResponse:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._post(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=await async_maybe_transform(body, model_post_params.ModelPostParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelPostResponse,
        )

    async def put(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        name: str,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ModelPutResponse:
        """
        Proxy requests to model entity inference endpoints.

        All inference requests must resolve to a `VirtualModel`. The platform's provider
        reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
        served model entity (named after the entity, with `default_model_entity` set to
        the entity ref) so this is the typical case; operators can also create custom
        VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
        which no VirtualModel can be found return `404`.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._put(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=await async_maybe_transform(body, model_put_params.ModelPutParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ModelPutResponse,
        )


class ModelResourceWithRawResponse:
    def __init__(self, model: ModelResource) -> None:
        self._model = model

        self.delete = to_raw_response_wrapper(
            model.delete,
        )
        self.get = to_raw_response_wrapper(
            model.get,
        )
        self.patch = to_raw_response_wrapper(
            model.patch,
        )
        self.post = to_raw_response_wrapper(
            model.post,
        )
        self.put = to_raw_response_wrapper(
            model.put,
        )


class AsyncModelResourceWithRawResponse:
    def __init__(self, model: AsyncModelResource) -> None:
        self._model = model

        self.delete = async_to_raw_response_wrapper(
            model.delete,
        )
        self.get = async_to_raw_response_wrapper(
            model.get,
        )
        self.patch = async_to_raw_response_wrapper(
            model.patch,
        )
        self.post = async_to_raw_response_wrapper(
            model.post,
        )
        self.put = async_to_raw_response_wrapper(
            model.put,
        )


class ModelResourceWithStreamingResponse:
    def __init__(self, model: ModelResource) -> None:
        self._model = model

        self.delete = to_streamed_response_wrapper(
            model.delete,
        )
        self.get = to_streamed_response_wrapper(
            model.get,
        )
        self.patch = to_streamed_response_wrapper(
            model.patch,
        )
        self.post = to_streamed_response_wrapper(
            model.post,
        )
        self.put = to_streamed_response_wrapper(
            model.put,
        )


class AsyncModelResourceWithStreamingResponse:
    def __init__(self, model: AsyncModelResource) -> None:
        self._model = model

        self.delete = async_to_streamed_response_wrapper(
            model.delete,
        )
        self.get = async_to_streamed_response_wrapper(
            model.get,
        )
        self.patch = async_to_streamed_response_wrapper(
            model.patch,
        )
        self.post = async_to_streamed_response_wrapper(
            model.post,
        )
        self.put = async_to_streamed_response_wrapper(
            model.put,
        )
