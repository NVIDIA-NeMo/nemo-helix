// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { AgentEvaluationNewRoute } from '@studio/routes/agents/AgentEvaluationNewRoute';
import { getAgentDetailRoute, getAgentRunEvaluationRoute } from '@studio/routes/utils';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen } from '@studio/tests/util/render';
import { http, HttpResponse } from 'msw';

const agentName = 'react-agent';
const workspace = workspace1.workspace;

describe('AgentEvaluationNewRoute', () => {
  it('sends an agent that cannot be evaluated back to its page', async () => {
    server.use(
      http.get('*/apis/agents/v2/workspaces/:workspace/agents/:name', () =>
        HttpResponse.json({ name: agentName, workspace, created_at: '2026-04-20T10:00:00Z' })
      )
    );
    renderRoute(undefined, {
      history: getAgentRunEvaluationRoute(workspace, agentName),
      routes: [
        { path: ROUTES.workspace.agentEvaluationNew, element: <AgentEvaluationNewRoute /> },
        { path: ROUTES.workspace.agentDetail, element: <LocationDisplay /> },
      ],
    });

    expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
      getAgentDetailRoute(workspace, agentName)
    );
  });
});
