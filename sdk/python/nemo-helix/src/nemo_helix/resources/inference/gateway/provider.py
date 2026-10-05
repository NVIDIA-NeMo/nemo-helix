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
from ....types.inference.gateway import provider_put_params, provider_post_params, provider_patch_params
from ....types.inference.gateway.provider_put_response import ProviderPutResponse
from ....types.inference.gateway.provider_post_response import ProviderPostResponse
from ....types.inference.gateway.provider_patch_response import ProviderPatchResponse
from ....types.inference.gateway.provider_ready_response import ProviderReadyResponse

__all__ = ["ProviderResource", "AsyncProviderResource"]


class ProviderResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> ProviderResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return ProviderResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ProviderResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return ProviderResourceWithStreamingResponse(self)

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
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
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
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
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
    ) -> ProviderPatchResponse:
        """
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=maybe_transform(body, provider_patch_params.ProviderPatchParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ProviderPatchResponse,
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
    ) -> ProviderPostResponse:
        """
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=maybe_transform(body, provider_post_params.ProviderPostParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ProviderPostResponse,
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
    ) -> ProviderPutResponse:
        """
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=maybe_transform(body, provider_put_params.ProviderPutParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ProviderPutResponse,
        )

    def ready(
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
    ) -> ProviderReadyResponse:
        """
        Check if a model provider is registered in the gateway's cache.

        This is a lightweight endpoint that only checks the gateway's internal state,
        without making any requests to the actual provider backend. Use this to verify
        the gateway is ready to route requests to a provider after deployment.

        Returns: 200 OK with provider info if the provider is registered 404 Not Found
        if the provider is not yet in the gateway's cache

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/ready",
                workspace=workspace,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ProviderReadyResponse,
        )


class AsyncProviderResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncProviderResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncProviderResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncProviderResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncProviderResourceWithStreamingResponse(self)

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
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
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
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
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
    ) -> ProviderPatchResponse:
        """
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=await async_maybe_transform(body, provider_patch_params.ProviderPatchParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ProviderPatchResponse,
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
    ) -> ProviderPostResponse:
        """
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=await async_maybe_transform(body, provider_post_params.ProviderPostParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ProviderPostResponse,
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
    ) -> ProviderPutResponse:
        """
        Proxy requests to provider inference endpoints.

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
                workspace=workspace,
                name=name,
                trailing_uri=trailing_uri,
            ),
            body=await async_maybe_transform(body, provider_put_params.ProviderPutParams),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ProviderPutResponse,
        )

    async def ready(
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
    ) -> ProviderReadyResponse:
        """
        Check if a model provider is registered in the gateway's cache.

        This is a lightweight endpoint that only checks the gateway's internal state,
        without making any requests to the actual provider backend. Use this to verify
        the gateway is ready to route requests to a provider after deployment.

        Returns: 200 OK with provider info if the provider is registered 404 Not Found
        if the provider is not yet in the gateway's cache

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
                "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/ready",
                workspace=workspace,
                name=name,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ProviderReadyResponse,
        )


class ProviderResourceWithRawResponse:
    def __init__(self, provider: ProviderResource) -> None:
        self._provider = provider

        self.delete = to_raw_response_wrapper(
            provider.delete,
        )
        self.get = to_raw_response_wrapper(
            provider.get,
        )
        self.patch = to_raw_response_wrapper(
            provider.patch,
        )
        self.post = to_raw_response_wrapper(
            provider.post,
        )
        self.put = to_raw_response_wrapper(
            provider.put,
        )
        self.ready = to_raw_response_wrapper(
            provider.ready,
        )


class AsyncProviderResourceWithRawResponse:
    def __init__(self, provider: AsyncProviderResource) -> None:
        self._provider = provider

        self.delete = async_to_raw_response_wrapper(
            provider.delete,
        )
        self.get = async_to_raw_response_wrapper(
            provider.get,
        )
        self.patch = async_to_raw_response_wrapper(
            provider.patch,
        )
        self.post = async_to_raw_response_wrapper(
            provider.post,
        )
        self.put = async_to_raw_response_wrapper(
            provider.put,
        )
        self.ready = async_to_raw_response_wrapper(
            provider.ready,
        )


class ProviderResourceWithStreamingResponse:
    def __init__(self, provider: ProviderResource) -> None:
        self._provider = provider

        self.delete = to_streamed_response_wrapper(
            provider.delete,
        )
        self.get = to_streamed_response_wrapper(
            provider.get,
        )
        self.patch = to_streamed_response_wrapper(
            provider.patch,
        )
        self.post = to_streamed_response_wrapper(
            provider.post,
        )
        self.put = to_streamed_response_wrapper(
            provider.put,
        )
        self.ready = to_streamed_response_wrapper(
            provider.ready,
        )


class AsyncProviderResourceWithStreamingResponse:
    def __init__(self, provider: AsyncProviderResource) -> None:
        self._provider = provider

        self.delete = async_to_streamed_response_wrapper(
            provider.delete,
        )
        self.get = async_to_streamed_response_wrapper(
            provider.get,
        )
        self.patch = async_to_streamed_response_wrapper(
            provider.patch,
        )
        self.post = async_to_streamed_response_wrapper(
            provider.post,
        )
        self.put = async_to_streamed_response_wrapper(
            provider.put,
        )
        self.ready = async_to_streamed_response_wrapper(
            provider.ready,
        )
