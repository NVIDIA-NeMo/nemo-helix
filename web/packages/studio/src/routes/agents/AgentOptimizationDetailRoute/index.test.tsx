// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { AgentOptimizationDetailRoute } from '@studio/routes/agents/AgentOptimizationDetailRoute';
import { getAgentOptimizationDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, within } from '@studio/tests/util/render';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const jobName = 'temperature-sweep';
const OPTIMIZE_JOB_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/optimize/:name`;

const renderStudy = (status: HelixJobStatus) => {
  server.use(
    http.get(OPTIMIZE_JOB_URL, () =>
      HttpResponse.json({
        name: jobName,
        workspace,
        status,
        spec: { agent: 'hermes', optimize_config: 'optimize.yaml' },
      })
    )
  );
  return renderRoute(undefined, {
    history: getAgentOptimizationDetailRoute(workspace, jobName),
    routes: [
      { path: ROUTES.workspace.agentOptimizationDetail, element: <AgentOptimizationDetailRoute /> },
    ],
  });
};

describe('AgentOptimizationDetailRoute', () => {
  it.each<[HelixJobStatus, string, string]>([
    ['created', 'Study queued', 'Waiting for the study to start. Trials appear once it finishes.'],
    ['pending', 'Study queued', 'Waiting for the study to start. Trials appear once it finishes.'],
    ['active', 'Study running', 'Trials appear once the study finishes.'],
  ])('shows a spinner while the study is %s', async (status, spinnerLabel, message) => {
    renderStudy(status);

    const inProgress = await screen.findByTestId('study-in-progress');
    expect(within(inProgress).getByLabelText(spinnerLabel)).toBeInTheDocument();
    expect(inProgress).toHaveTextContent(message);
  });
});
