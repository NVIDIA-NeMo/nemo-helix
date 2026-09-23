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

from typing import Any, Dict, Union, cast
from typing_extensions import Literal, overload

import httpx

from ..._types import Body, Omit, Query, Headers, NoneType, NotGiven, omit, not_given
from ..._utils import path_template, required_args, maybe_transform, async_maybe_transform
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
from ...types.intake import AnnotationSortField, annotation_list_params, annotation_create_params
from ...types.intake.annotation import Annotation
from ...types.intake.annotation_sort_field import AnnotationSortField
from ...types.intake.annotation_filter_param import AnnotationFilterParam
from ..._exceptions import ConflictError

__all__ = ["AnnotationsResource", "AsyncAnnotationsResource"]


class AnnotationsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AnnotationsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AnnotationsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AnnotationsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AnnotationsResourceWithStreamingResponse(self)

    @overload
    def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["feedback"],
        session_id: str,
        value: Literal["positive", "negative"],
        span_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """Create Annotation

        Args:
          kind: Discriminator.

        Always `feedback` for this variant.

          session_id: Id of the session this annotation belongs to. Always required.

          value: Sentiment of the feedback.

          span_id: Id of the span this annotation applies to. Omit to annotate the whole session
              instead of a specific span.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["note"],
        session_id: str,
        text: str,
        span_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """Create Annotation

        Args:
          kind: Discriminator.

        Always `note` for this variant.

          session_id: Id of the session this annotation belongs to. Always required.

          text: The note content. 1 to 10,000 characters.

          span_id: Id of the span this annotation applies to. Omit to annotate the whole session
              instead of a specific span.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["metadata"],
        metadata: Dict[str, object],
        session_id: str,
        span_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """Create Annotation

        Args:
          kind: Discriminator.

        Always `metadata` for this variant.

          metadata: Arbitrary key/value pairs. Must contain at least one entry.

          session_id: Id of the session this annotation belongs to. Always required.

          span_id: Id of the span this annotation applies to. Omit to annotate the whole session
              instead of a specific span.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["label"],
        session_id: str,
        value: Union[str, float],
        value_type: Literal["text", "numeric"],
        name: str | Omit = omit,
        span_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """Create Annotation

        Args:
          kind: Discriminator.

        Always `label` for this variant.

          session_id: Id of the session this annotation belongs to. Always required.

          value: The label's value. Must be a string when `value_type=text` and a number when
              `value_type=numeric`.

          value_type: Whether `value` should be interpreted as text (`text`) or a number (`numeric`).

          name: Name identifying what the label measures (e.g., `severity`, `helpfulness`).
              Optional for text labels; required for numeric labels.

          span_id: Id of the span this annotation applies to. Omit to annotate the whole session
              instead of a specific span.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @required_args(
        ["kind", "session_id", "value"],
        ["kind", "session_id", "text"],
        ["kind", "metadata", "session_id"],
        ["kind", "session_id", "value", "value_type"],
    )
    def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["feedback"] | Literal["note"] | Literal["metadata"] | Literal["label"],
        session_id: str,
        value: Literal["positive", "negative"] | Union[str, float] | Omit = omit,
        span_id: str | Omit = omit,
        text: str | Omit = omit,
        metadata: Dict[str, object] | Omit = omit,
        value_type: Literal["text", "numeric"] | Omit = omit,
        name: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        try:
            if workspace is None:
                workspace = self._client._get_workspace_path_param()
            if not workspace:
                raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
            return cast(
                Annotation,
                self._post(
                    path_template("/apis/intake/v2/workspaces/{workspace}/annotations", workspace=workspace),
                    body=maybe_transform(
                        {
                            "kind": kind,
                            "session_id": session_id,
                            "value": value,
                            "span_id": span_id,
                            "text": text,
                            "metadata": metadata,
                            "value_type": value_type,
                            "name": name,
                        },
                        annotation_create_params.AnnotationCreateParams,
                    ),
                    options=make_request_options(
                        extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                    ),
                    cast_to=cast(Any, Annotation),  # Union types cannot be passed in as arguments in the type system
                ),
            )
        except ConflictError:
            if not exist_ok:
                raise
            return self.retrieve(name = name, workspace = workspace)

    def retrieve(
        self,
        annotation_id: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """
        Get Annotation

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
        if not annotation_id:
            raise ValueError(f"Expected a non-empty value for `annotation_id` but received {annotation_id!r}")
        return cast(
            Annotation,
            self._get(
                path_template(
                    "/apis/intake/v2/workspaces/{workspace}/annotations/{annotation_id}",
                    workspace=workspace,
                    annotation_id=annotation_id,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=cast(Any, Annotation),  # Union types cannot be passed in as arguments in the type system
            ),
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: AnnotationFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: AnnotationSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SyncDefaultPagination[Annotation]:
        """
        List Annotations

        Args:
          filter: Filter annotations by span_id, session_id, kind, name, created_by, and
              created_at range.

          page: Page number.

          page_size: Page size.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/annotations", workspace=workspace),
            page=SyncDefaultPagination[Annotation],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    annotation_list_params.AnnotationListParams,
                ),
            ),
            model=cast(Any, Annotation),  # Union types cannot be passed in as arguments in the type system
        )

    def delete(
        self,
        annotation_id: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Delete Annotation

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
        if not annotation_id:
            raise ValueError(f"Expected a non-empty value for `annotation_id` but received {annotation_id!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return self._delete(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/annotations/{annotation_id}",
                workspace=workspace,
                annotation_id=annotation_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class AsyncAnnotationsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncAnnotationsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#accessing-raw-response-data-e-g-headers
        """
        return AsyncAnnotationsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncAnnotationsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#with_streaming_response
        """
        return AsyncAnnotationsResourceWithStreamingResponse(self)

    @overload
    async def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["feedback"],
        session_id: str,
        value: Literal["positive", "negative"],
        span_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """Create Annotation

        Args:
          kind: Discriminator.

        Always `feedback` for this variant.

          session_id: Id of the session this annotation belongs to. Always required.

          value: Sentiment of the feedback.

          span_id: Id of the span this annotation applies to. Omit to annotate the whole session
              instead of a specific span.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    async def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["note"],
        session_id: str,
        text: str,
        span_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """Create Annotation

        Args:
          kind: Discriminator.

        Always `note` for this variant.

          session_id: Id of the session this annotation belongs to. Always required.

          text: The note content. 1 to 10,000 characters.

          span_id: Id of the span this annotation applies to. Omit to annotate the whole session
              instead of a specific span.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    async def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["metadata"],
        metadata: Dict[str, object],
        session_id: str,
        span_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """Create Annotation

        Args:
          kind: Discriminator.

        Always `metadata` for this variant.

          metadata: Arbitrary key/value pairs. Must contain at least one entry.

          session_id: Id of the session this annotation belongs to. Always required.

          span_id: Id of the span this annotation applies to. Omit to annotate the whole session
              instead of a specific span.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    async def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["label"],
        session_id: str,
        value: Union[str, float],
        value_type: Literal["text", "numeric"],
        name: str | Omit = omit,
        span_id: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """Create Annotation

        Args:
          kind: Discriminator.

        Always `label` for this variant.

          session_id: Id of the session this annotation belongs to. Always required.

          value: The label's value. Must be a string when `value_type=text` and a number when
              `value_type=numeric`.

          value_type: Whether `value` should be interpreted as text (`text`) or a number (`numeric`).

          name: Name identifying what the label measures (e.g., `severity`, `helpfulness`).
              Optional for text labels; required for numeric labels.

          span_id: Id of the span this annotation applies to. Omit to annotate the whole session
              instead of a specific span.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @required_args(
        ["kind", "session_id", "value"],
        ["kind", "session_id", "text"],
        ["kind", "metadata", "session_id"],
        ["kind", "session_id", "value", "value_type"],
    )
    async def create(
        self,
        *,
        workspace: str | None = None,
        kind: Literal["feedback"] | Literal["note"] | Literal["metadata"] | Literal["label"],
        session_id: str,
        value: Literal["positive", "negative"] | Union[str, float] | Omit = omit,
        span_id: str | Omit = omit,
        text: str | Omit = omit,
        metadata: Dict[str, object] | Omit = omit,
        value_type: Literal["text", "numeric"] | Omit = omit,
        name: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        exist_ok: bool = False,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        try:
            if workspace is None:
                workspace = self._client._get_workspace_path_param()
            if not workspace:
                raise ValueError(f"Expected a non-empty value for `workspace` but received {workspace!r}")
            return cast(
                Annotation,
                await self._post(
                    path_template("/apis/intake/v2/workspaces/{workspace}/annotations", workspace=workspace),
                    body=await async_maybe_transform(
                        {
                            "kind": kind,
                            "session_id": session_id,
                            "value": value,
                            "span_id": span_id,
                            "text": text,
                            "metadata": metadata,
                            "value_type": value_type,
                            "name": name,
                        },
                        annotation_create_params.AnnotationCreateParams,
                    ),
                    options=make_request_options(
                        extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                    ),
                    cast_to=cast(Any, Annotation),  # Union types cannot be passed in as arguments in the type system
                ),
            )
        except ConflictError:
            if not exist_ok:
                raise
            return await self.retrieve(name = name, workspace = workspace)

    async def retrieve(
        self,
        annotation_id: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Annotation:
        """
        Get Annotation

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
        if not annotation_id:
            raise ValueError(f"Expected a non-empty value for `annotation_id` but received {annotation_id!r}")
        return cast(
            Annotation,
            await self._get(
                path_template(
                    "/apis/intake/v2/workspaces/{workspace}/annotations/{annotation_id}",
                    workspace=workspace,
                    annotation_id=annotation_id,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=cast(Any, Annotation),  # Union types cannot be passed in as arguments in the type system
            ),
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: AnnotationFilterParam | Omit = omit,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        sort: AnnotationSortField | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AsyncPaginator[Annotation, AsyncDefaultPagination[Annotation]]:
        """
        List Annotations

        Args:
          filter: Filter annotations by span_id, session_id, kind, name, created_by, and
              created_at range.

          page: Page number.

          page_size: Page size.

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
            path_template("/apis/intake/v2/workspaces/{workspace}/annotations", workspace=workspace),
            page=AsyncDefaultPagination[Annotation],
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "filter": filter,
                        "page": page,
                        "page_size": page_size,
                        "sort": sort,
                    },
                    annotation_list_params.AnnotationListParams,
                ),
            ),
            model=cast(Any, Annotation),  # Union types cannot be passed in as arguments in the type system
        )

    async def delete(
        self,
        annotation_id: str,
        *,
        workspace: str | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> None:
        """
        Delete Annotation

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
        if not annotation_id:
            raise ValueError(f"Expected a non-empty value for `annotation_id` but received {annotation_id!r}")
        extra_headers = {"Accept": "*/*", **(extra_headers or {})}
        return await self._delete(
            path_template(
                "/apis/intake/v2/workspaces/{workspace}/annotations/{annotation_id}",
                workspace=workspace,
                annotation_id=annotation_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=NoneType,
        )


class AnnotationsResourceWithRawResponse:
    def __init__(self, annotations: AnnotationsResource) -> None:
        self._annotations = annotations

        self.create = to_raw_response_wrapper(
            annotations.create,
        )
        self.retrieve = to_raw_response_wrapper(
            annotations.retrieve,
        )
        self.list = to_raw_response_wrapper(
            annotations.list,
        )
        self.delete = to_raw_response_wrapper(
            annotations.delete,
        )


class AsyncAnnotationsResourceWithRawResponse:
    def __init__(self, annotations: AsyncAnnotationsResource) -> None:
        self._annotations = annotations

        self.create = async_to_raw_response_wrapper(
            annotations.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            annotations.retrieve,
        )
        self.list = async_to_raw_response_wrapper(
            annotations.list,
        )
        self.delete = async_to_raw_response_wrapper(
            annotations.delete,
        )


class AnnotationsResourceWithStreamingResponse:
    def __init__(self, annotations: AnnotationsResource) -> None:
        self._annotations = annotations

        self.create = to_streamed_response_wrapper(
            annotations.create,
        )
        self.retrieve = to_streamed_response_wrapper(
            annotations.retrieve,
        )
        self.list = to_streamed_response_wrapper(
            annotations.list,
        )
        self.delete = to_streamed_response_wrapper(
            annotations.delete,
        )


class AsyncAnnotationsResourceWithStreamingResponse:
    def __init__(self, annotations: AsyncAnnotationsResource) -> None:
        self._annotations = annotations

        self.create = async_to_streamed_response_wrapper(
            annotations.create,
        )
        self.retrieve = async_to_streamed_response_wrapper(
            annotations.retrieve,
        )
        self.list = async_to_streamed_response_wrapper(
            annotations.list,
        )
        self.delete = async_to_streamed_response_wrapper(
            annotations.delete,
        )
