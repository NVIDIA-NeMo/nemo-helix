// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import { SuspenseOutlet } from '@studio/components/Layouts/SuspenseOutlet';
import { isLocalDevelopmentEnv, REACT_QUERY_DEVTOOLS_ENABLED } from '@studio/constants/environment';
import { BreadcrumbsProvider } from '@studio/providers/breadcrumbs/BreadcrumbsProvider';
import { WorkersProvider } from '@studio/providers/workers/WorkersProvider';
import { WorkspaceProvider } from '@studio/providers/workspace';
import { Suspense, lazy } from 'react';

const ReactQueryDevtools =
  isLocalDevelopmentEnv && REACT_QUERY_DEVTOOLS_ENABLED
    ? lazy(() =>
        // Lazy load in development
        import('@tanstack/react-query-devtools').then((res) => ({
          default: res.ReactQueryDevtools,
        }))
      )
    : () => null;

export const RootLayout = () => {
  return (
    <WorkspaceProvider>
      <ToastProvider>
        <WorkersProvider>
          <BreadcrumbsProvider>
            <SuspenseOutlet />
            <Suspense>
              <ReactQueryDevtools initialIsOpen={false} />
            </Suspense>
          </BreadcrumbsProvider>
        </WorkersProvider>
      </ToastProvider>
    </WorkspaceProvider>
  );
};
