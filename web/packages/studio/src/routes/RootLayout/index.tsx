// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import { Loading } from '@studio/components/Layouts/Loading';
import { isLocalDevelopmentEnv, REACT_QUERY_DEVTOOLS_ENABLED } from '@studio/constants/environment';
import { BreadcrumbsProvider } from '@studio/providers/breadcrumbs/BreadcrumbsProvider';
import { WorkersProvider } from '@studio/providers/workers/WorkersProvider';
import { WorkspaceProvider } from '@studio/providers/workspace';
import { PlatformGuard } from '@studio/routes/RootLayout/PlatformGuard';
import { Suspense, lazy } from 'react';
import { Outlet } from 'react-router';

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
    <PlatformGuard>
      <WorkspaceProvider>
        <ToastProvider>
          <WorkersProvider>
            <BreadcrumbsProvider>
              <Suspense fallback={<Loading />}>
                <Outlet />
                <ReactQueryDevtools initialIsOpen={false} />
              </Suspense>
            </BreadcrumbsProvider>
          </WorkersProvider>
        </ToastProvider>
      </WorkspaceProvider>
    </PlatformGuard>
  );
};
