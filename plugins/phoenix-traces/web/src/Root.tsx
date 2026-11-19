// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Navigate, Route, Routes } from 'react-router';
import { pluginPath } from './paths';
import { TraceDetailPage } from './TraceDetailPage';
import { TracesPage } from './TracesPage';
import type { PluginRootProps } from './types';

/** Phoenix trace viewer. Rendered inside Studio's React tree (router, query client, theme). */
export function Root({ host }: PluginRootProps) {
  return (
    <Routes>
      <Route index element={<Navigate to={pluginPath(host.workspaceId, 'traces')} replace />} />
      <Route path="traces" element={<TracesPage host={host} />} />
      <Route path="traces/:traceId" element={<TraceDetailPage host={host} />} />
      <Route path="*" element={<Navigate to={pluginPath(host.workspaceId, 'traces')} replace />} />
    </Routes>
  );
}
