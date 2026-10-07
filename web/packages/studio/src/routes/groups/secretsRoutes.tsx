// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ROUTES } from '@studio/constants/routes';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const SecretsListRoute = lazy(() =>
  import('@studio/routes/SecretsListRoute').then((module) => ({ default: module.SecretsListRoute }))
);

export const secretsRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.secrets,
    element: <SecretsListRoute />,
    errorElement: <RouteErrorPanel title="Secrets" />,
  },
];
