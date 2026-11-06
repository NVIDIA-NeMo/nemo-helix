// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { queryClientConfig } from '@studio/api/queryClient';
import { QueryClient, type QueryClientConfig, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

const testQueryClientConfig: QueryClientConfig = {
  ...queryClientConfig,
  defaultOptions: {
    ...queryClientConfig.defaultOptions,
    queries: {
      ...queryClientConfig.defaultOptions?.queries,
      retry: false,
    },
  },
};

export class TestQueryClient extends QueryClient {
  constructor(options?: QueryClientConfig) {
    super({
      ...testQueryClientConfig,
      ...options,
    });
  }
}

// This is meant to be used in hook tests that use renderHook(fn, { wrapper })
export const wrapper = ({ children }: { children?: React.ReactNode }) => {
  const queryClient = new TestQueryClient();
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
};
