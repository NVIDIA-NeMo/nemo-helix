// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_INTAKE_ENABLED', 'true');
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'false');
});

import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';

describe('Agent overview with the optimizer flag off', () => {
  it('hides the open insights panel but keeps trace statistics', async () => {
    renderRoute(undefined, {
      history: getAgentDetailRoute(workspace1.workspace, 'react-agent'),
      routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
    });

    expect(await screen.findByText('Trace statistics')).toBeInTheDocument();
    expect(screen.queryByText('Open insights')).not.toBeInTheDocument();
  });
});
