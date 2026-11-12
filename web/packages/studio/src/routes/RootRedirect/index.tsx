// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFallbackWorkspace } from '@studio/hooks/useFallbackWorkspace';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Navigate } from 'react-router';

export const RootRedirect: FC = () => {
  const workspace = useFallbackWorkspace();
  return workspace ? <Navigate to={getWorkspaceDetailsDefaultRoute(workspace)} replace /> : null;
};
