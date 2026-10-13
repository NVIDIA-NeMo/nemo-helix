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
const OPTIMIZE_JOB_URL = `${PLATFORM_BASE_URL}/apis/agent-optimization/v2/workspaces/:workspace/jobs/run-strategy/:name`;
const OPTIMIZE_JOB_STATUS_URL = `${OPTIMIZE_JOB_URL}/status`;

const step = (name: string, status: HelixJobStatus) => ({
  id: `step-${name}`,
  name,
  status,
  status_details: {},
  error_details: {},
  tasks: [],
  created_at: '2026-08-13T09:00:00Z',
  updated_at: '2026-08-13T09:30:00Z',
});

const renderStudy = (status: HelixJobStatus) => {
  server.use(
    http.get(`${OPTIMIZE_JOB_URL}/results`, () => HttpResponse.json({ data: [] })),
    http.get(OPTIMIZE_JOB_URL, () =>
      HttpResponse.json({
        name: jobName,
        workspace,
        status,
        created_at: '2026-08-13T09:00:00Z',
        spec: {
          strategy: 'legacy',
          agent: 'hermes',
          optimize_config: 'optimize.yaml',
          optimize_config_fileset: 'hermes-bundle',
        },
      })
    ),
    http.get(OPTIMIZE_JOB_STATUS_URL, () =>
      HttpResponse.json({
        id: 'opt-1',
        name: jobName,
        status,
        status_details: {},
        error_details: {},
        steps: [step('prepare', 'completed'), step('optimize', status)],
        created_at: '2026-08-13T09:00:00Z',
        updated_at: '2026-08-13T09:30:00Z',
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
  ])('shows a header spinner while the study is %s', async (status, spinnerLabel, message) => {
    renderStudy(status);

    const inProgress = await screen.findByTestId('study-in-progress');
    expect(within(inProgress).queryByLabelText(spinnerLabel)).not.toBeInTheDocument();
    expect(screen.getByLabelText(spinnerLabel)).toBeInTheDocument();
    expect(within(inProgress).getByText(message)).toHaveAttribute('role', 'status');
  });

  it('shows the study spec, step progress, and logs while running', async () => {
    renderStudy('active');

    const inProgress = await screen.findByTestId('study-in-progress');
    expect(within(inProgress).getByText('legacy')).toBeInTheDocument();
    expect(within(inProgress).getByText('hermes')).toBeInTheDocument();
    expect(within(inProgress).getByText('optimize.yaml')).toBeInTheDocument();
    expect(within(inProgress).getByText('hermes-bundle')).toBeInTheDocument();

    const steps = await within(inProgress).findAllByRole('listitem');
    expect(steps).toHaveLength(2);
    expect(within(steps[0]).getByText('prepare')).toBeInTheDocument();
    expect(within(steps[1]).getByText('optimize')).toBeInTheDocument();

    expect(screen.getAllByText('Logs')).toHaveLength(1);
    expect(await screen.findByText('No artifacts yet')).toBeInTheDocument();
  });

  it.each<HelixJobStatus>(['completed', 'error', 'cancelled'])(
    'keeps logs and generated artifacts visible when the job is %s',
    async (status) => {
      renderStudy(status);
      server.use(
        http.get(`${OPTIMIZE_JOB_URL}/results`, () =>
          HttpResponse.json({
            data: [
              {
                name: 'switchyard',
                artifact_storage_type: 'fileset',
                artifact_url: `fileset://${workspace}/routing-artifacts#results/switchyard`,
              },
            ],
          })
        ),
        http.get(
          `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/routing-artifacts/files`,
          () =>
            HttpResponse.json({
              data: [
                {
                  path: 'results/switchyard/agent-routed.yaml',
                  size: 128,
                  file_ref: 'routed-config',
                },
              ],
            })
        ),
        http.get(`${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name/logs`, () =>
          HttpResponse.json({
            data: [
              {
                timestamp: '2026-08-13T09:30:00Z',
                level: 'INFO',
                message: 'Created routed virtual model',
              },
            ],
            total: 1,
            next_page: '',
            prev_page: '',
          })
        )
      );

      expect(await screen.findByText('results/switchyard/agent-routed.yaml')).toBeInTheDocument();
      expect(await screen.findByText(/Created routed virtual model/)).toBeInTheDocument();
      expect(screen.getAllByText('Logs')).toHaveLength(1);
      expect(screen.getByText('Artifacts')).toBeInTheDocument();
    }
  );
});
