/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { isNonRetryableClientError } from '@nemo/common/src/api/common/utils';
import { isLocalDevelopmentEnv } from '@studio/constants/environment';
import { QueryClient, QueryClientConfig } from '@tanstack/react-query';

declare global {
  interface Window {
    /** Read by the TanStack Query browser devtools extensions. */
    __TANSTACK_QUERY_CLIENT__?: QueryClient;
  }
}

/**
 * The default number of times to retry a failed query,
 * 3 is the [TanStack Query default](https://tanstack.com/query/v4/docs/react/guides/query-retries)
 */
export const DEFAULT_QUERY_RETRY_COUNT = 3;

export const queryClientConfig: QueryClientConfig = {
  defaultOptions: {
    queries: {
      // A 404 or 403 won't change on retry; failing fast lets not-found and error states show immediately.
      retry: (failureCount, error) =>
        !isNonRetryableClientError(error) && failureCount < DEFAULT_QUERY_RETRY_COUNT,
      // With SSR, we usually want to set some default staleTime
      // above 0 to avoid refetching immediately on the client
      staleTime: 60 * 1000,
      refetchOnWindowFocus: false,
    },
  },
};

/**
 * The global app query client, used by every useQuery and useMutation
 */
export const queryClient = new QueryClient(queryClientConfig);

if (isLocalDevelopmentEnv) {
  window.__TANSTACK_QUERY_CLIENT__ = queryClient;
}
