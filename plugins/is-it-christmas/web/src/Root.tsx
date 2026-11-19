// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Navigate, Route, Routes } from 'react-router';
import { IsItChristmasPage } from './IsItChristmasPage';
import { pluginPath } from './paths';
import type { PluginRootProps } from './types';

/** Rendered inside Studio's React tree (shared router, query client, theme). */
export function Root({ host }: PluginRootProps) {
  return (
    <Routes>
      <Route index element={<IsItChristmasPage host={host} />} />
      <Route path="*" element={<Navigate to={pluginPath(host.workspaceId)} replace />} />
    </Routes>
  );
}
