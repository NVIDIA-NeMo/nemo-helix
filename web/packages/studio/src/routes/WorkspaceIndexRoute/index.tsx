// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Navigate } from 'react-router';

/**
 * Just redirects to workspace default route
 */
export const WorkspaceIndexRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  return <Navigate to={getWorkspaceDetailsDefaultRoute(workspace)} replace />;
};
