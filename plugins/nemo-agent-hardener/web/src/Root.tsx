// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


import { configureClient } from '@agent-hardener/api/fetcher';
import { HostProvider } from '@agent-hardener/host';
import { ROUTE_PATHS } from '@agent-hardener/paths';
import { AgentHardenerManifestDetailRoute } from '@agent-hardener/routes/AgentHardenerManifestDetailRoute';
import { AgentHardenerManifestListRoute } from '@agent-hardener/routes/AgentHardenerManifestListRoute';
import { AgentHardenerRunDetailsRoute } from '@agent-hardener/routes/AgentHardenerRunDetailsRoute';
import { AgentHardenerRunListRoute } from '@agent-hardener/routes/AgentHardenerRunListRoute';
import { NewAgentHardenerManifestRoute } from '@agent-hardener/routes/NewAgentHardenerManifestRoute';
import type { PluginRootProps } from '@agent-hardener/types';
import { Route, Routes } from 'react-router';

/**
 * Agent Hardener plugin root.
 *
 * Studio renders this inside its own React tree — same Router, QueryClient, and
 * KaizenThemeProvider — so the routes below are plain `<Route>`s relative to the
 * plugin's mount point and the tables come from Studio's shared `@nemo/common`.
 *
 * The generated agent-hardener client is the plugin's own (Studio's SDK only covers
 * platform services), so it is pointed at the host's token getter here. This runs
 * during render rather than in an effect because child routes issue requests on
 * their first render, before an effect would have fired.
 */
export function Root({ host }: PluginRootProps) {
  configureClient({ getAccessToken: host.auth.getAccessToken, baseUrl: host.apiBaseUrl });

  return (
    <HostProvider host={host}>
      <Routes>
        <Route path={ROUTE_PATHS.runList} element={<AgentHardenerRunListRoute />} />
        <Route path={ROUTE_PATHS.manifestList} element={<AgentHardenerManifestListRoute />} />
        <Route path={ROUTE_PATHS.manifestNew} element={<NewAgentHardenerManifestRoute />} />
        <Route path={ROUTE_PATHS.manifestDetail} element={<AgentHardenerManifestDetailRoute />} />
        <Route path={ROUTE_PATHS.runDetails} element={<AgentHardenerRunDetailsRoute />} />
      </Routes>
    </HostProvider>
  );
}
