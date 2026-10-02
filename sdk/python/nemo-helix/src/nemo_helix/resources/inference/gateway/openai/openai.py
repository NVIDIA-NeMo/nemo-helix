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

from .v1.v1 import (
    V1Resource,
    AsyncV1Resource,
    V1ResourceWithRawResponse,
    AsyncV1ResourceWithRawResponse,
    V1ResourceWithStreamingResponse,
    AsyncV1ResourceWithStreamingResponse,
)
from ....._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ....._utils import path_template, maybe_transform, async_maybe_transform
from ....._compat import cached_property
from ....._resource import SyncAPIResource, AsyncAPIResource
from ....._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ....._base_client import make_request_options
from .....types.inference.gateway import openai_put_params, openai_post_params, openai_patch_params
from .....types.inference.gateway.openai_put_response import OpenAIPutResponse
from .....types.inference.gateway.openai_post_response import OpenAIPostResponse
from .....types.inference.gateway.openai_patch_response import OpenAIPatchResponse

__all__ = ["OpenAIResource", "AsyncOpenAIResource"]


class OpenAIResource(SyncAPIResource):
    @cached_property
    def v1(self) -> V1Resource:
        return V1Resource(self._client)

    @cached_property
    def with_raw_response(self) -> OpenAIResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return OpenAIResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> OpenAIResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return OpenAIResourceWithStreamingResponse(self)

    def delete(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> object:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._delete(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
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
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> object:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._get(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
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
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> OpenAIPatchResponse:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._patch(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
                trailing_uri=trailing_uri,
            ),
            body=maybe_transform(body, openai_patch_params.OpenAIPatchParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=OpenAIPatchResponse,
        )

    def post(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> OpenAIPostResponse:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._post(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
                trailing_uri=trailing_uri,
            ),
            body=maybe_transform(body, openai_post_params.OpenAIPostParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=OpenAIPostResponse,
        )

    def put(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> OpenAIPutResponse:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return self._put(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
                trailing_uri=trailing_uri,
            ),
            body=maybe_transform(body, openai_put_params.OpenAIPutParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=OpenAIPutResponse,
        )


class AsyncOpenAIResource(AsyncAPIResource):
    @cached_property
    def v1(self) -> AsyncV1Resource:
        return AsyncV1Resource(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncOpenAIResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncOpenAIResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncOpenAIResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncOpenAIResourceWithStreamingResponse(self)

    async def delete(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> object:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._delete(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
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
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> object:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._get(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
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
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> OpenAIPatchResponse:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._patch(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
                trailing_uri=trailing_uri,
            ),
            body=await async_maybe_transform(body, openai_patch_params.OpenAIPatchParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=OpenAIPatchResponse,
        )

    async def post(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> OpenAIPostResponse:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._post(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
                trailing_uri=trailing_uri,
            ),
            body=await async_maybe_transform(body, openai_post_params.OpenAIPostParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=OpenAIPostResponse,
        )

    async def put(
        self,
        trailing_uri: str,
        *,
        workspace: str | None = None,
        body: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> OpenAIPutResponse:
        """
        Proxy requests to OpenAI-compatible inference endpoints.

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
        if not trailing_uri:
            raise ValueError(f"Expected a non-empty value for `trailing_uri` but received {trailing_uri!r}")
        return await self._put(
            path_template(
                "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
                workspace=workspace,
                trailing_uri=trailing_uri,
            ),
            body=await async_maybe_transform(body, openai_put_params.OpenAIPutParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=OpenAIPutResponse,
        )


class OpenAIResourceWithRawResponse:
    def __init__(self, openai: OpenAIResource) -> None:
        self._openai = openai

        self.delete = to_raw_response_wrapper(
            openai.delete,
        )
        self.get = to_raw_response_wrapper(
            openai.get,
        )
        self.patch = to_raw_response_wrapper(
            openai.patch,
        )
        self.post = to_raw_response_wrapper(
            openai.post,
        )
        self.put = to_raw_response_wrapper(
            openai.put,
        )

    @cached_property
    def v1(self) -> V1ResourceWithRawResponse:
        return V1ResourceWithRawResponse(self._openai.v1)


class AsyncOpenAIResourceWithRawResponse:
    def __init__(self, openai: AsyncOpenAIResource) -> None:
        self._openai = openai

        self.delete = async_to_raw_response_wrapper(
            openai.delete,
        )
        self.get = async_to_raw_response_wrapper(
            openai.get,
        )
        self.patch = async_to_raw_response_wrapper(
            openai.patch,
        )
        self.post = async_to_raw_response_wrapper(
            openai.post,
        )
        self.put = async_to_raw_response_wrapper(
            openai.put,
        )

    @cached_property
    def v1(self) -> AsyncV1ResourceWithRawResponse:
        return AsyncV1ResourceWithRawResponse(self._openai.v1)


class OpenAIResourceWithStreamingResponse:
    def __init__(self, openai: OpenAIResource) -> None:
        self._openai = openai

        self.delete = to_streamed_response_wrapper(
            openai.delete,
        )
        self.get = to_streamed_response_wrapper(
            openai.get,
        )
        self.patch = to_streamed_response_wrapper(
            openai.patch,
        )
        self.post = to_streamed_response_wrapper(
            openai.post,
        )
        self.put = to_streamed_response_wrapper(
            openai.put,
        )

    @cached_property
    def v1(self) -> V1ResourceWithStreamingResponse:
        return V1ResourceWithStreamingResponse(self._openai.v1)


class AsyncOpenAIResourceWithStreamingResponse:
    def __init__(self, openai: AsyncOpenAIResource) -> None:
        self._openai = openai

        self.delete = async_to_streamed_response_wrapper(
            openai.delete,
        )
        self.get = async_to_streamed_response_wrapper(
            openai.get,
        )
        self.patch = async_to_streamed_response_wrapper(
            openai.patch,
        )
        self.post = async_to_streamed_response_wrapper(
            openai.post,
        )
        self.put = async_to_streamed_response_wrapper(
            openai.put,
        )

    @cached_property
    def v1(self) -> AsyncV1ResourceWithStreamingResponse:
        return AsyncV1ResourceWithStreamingResponse(self._openai.v1)
