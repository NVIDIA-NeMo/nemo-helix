// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MockToastProvider } from '@nemo/common/src/tests/MockToastProvider';
import {
  ThemeProvider as KaizenThemeProvider,
  TooltipProvider,
} from '@nvidia/foundations-react-core';
import { BreadcrumbsProvider } from '@studio/providers/breadcrumbs/BreadcrumbsProvider';
import { WorkersProvider } from '@studio/providers/workers/WorkersProvider';
import { MockWorkspaceProvider } from '@studio/tests/mocks/MockWorkspaceProvider';
import { TestQueryClient } from '@studio/tests/util/TestQueryClient';
import { QueryClientConfig, QueryClientProvider } from '@tanstack/react-query';
import { PropsWithChildren, StrictMode, Suspense } from 'react';

export interface TestProvidersOptions {
  queryClientConfig?: QueryClientConfig;
}

interface Props {
  options?: TestProvidersOptions;
}

export const TestProviders = ({ options, children }: PropsWithChildren<Props>) => {
  const queryClient = new TestQueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
    ...options?.queryClientConfig,
  });

  return (
    <StrictMode>
      <KaizenThemeProvider className="h-full" density="standard" theme="light">
        <QueryClientProvider client={queryClient}>
          <TooltipProvider>
            <MockToastProvider>
              <MockWorkspaceProvider>
                <BreadcrumbsProvider>
                  <WorkersProvider>
                    <Suspense fallback={<div>Loading...</div>}>{children}</Suspense>
                  </WorkersProvider>
                </BreadcrumbsProvider>
              </MockWorkspaceProvider>
            </MockToastProvider>
          </TooltipProvider>
        </QueryClientProvider>
      </KaizenThemeProvider>
    </StrictMode>
  );
};
