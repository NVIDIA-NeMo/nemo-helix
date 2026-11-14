// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ROUTES } from '@studio/constants/routes';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const InferenceProvidersListRoute = lazy(() =>
  import('@studio/routes/InferenceProvidersListRoute').then((module) => ({
    default: module.InferenceProvidersListRoute,
  }))
);

export const inferenceProviderRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.inferenceProviders,
    element: <InferenceProvidersListRoute />,
    errorElement: <RouteErrorPanel title="Inference Providers" />,
  },
];
