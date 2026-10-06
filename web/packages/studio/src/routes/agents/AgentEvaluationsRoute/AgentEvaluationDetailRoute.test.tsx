// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { AgentEvaluationDetailRoute } from '@studio/routes/agents/AgentEvaluationsRoute';
import { getAgentEvaluationDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const JOB_NAME = 'eval-gym-run';

const completedJob = {
  name: JOB_NAME,
  workspace,
  status: 'completed',
  created_at: '2026-08-01T00:00:00Z',
  updated_at: '2026-08-01T00:05:00Z',
  spec: { target: { kind: 'agent', agent: { name: 'my-agent' } }, tasks: [{}] },
};

const renderDetail = () =>
  renderRoute(<AgentEvaluationDetailRoute />, {
    history: getAgentEvaluationDetailRoute(workspace, JOB_NAME),
    routes: [
      { path: ROUTES.workspace.agentEvaluationDetail, element: <AgentEvaluationDetailRoute /> },
    ],
  });

describe('AgentEvaluationDetailRoute', () => {
  it('renders the not-found state when the job lookup returns null', async () => {
    // Default MSW handler returns ``{ data: [] }`` for the list endpoint;
    // there's no handler for the single-job GET, so MSW falls through to a
    // 404 → fetchAgentEvalJob resolves to null → not-found UI.
    renderRoute(<AgentEvaluationDetailRoute />, {
      history: getAgentEvaluationDetailRoute(workspace, 'eval-missing'),
      routes: [
        { path: ROUTES.workspace.agentEvaluationDetail, element: <AgentEvaluationDetailRoute /> },
      ],
    });
    expect(await screen.findByText('Evaluation not found')).toBeInTheDocument();
  });

  describe('run outcome', () => {
    const failedJob = {
      ...completedJob,
      status: 'error',
      error_details: { message: 'Job exited with code 1' },
      status_details: {
        message: 'Job exited with code 1',
        evaluation: {
          unit: 'trials',
          total: 3,
          errored: 3,
          scored: 0,
          failed: true,
          message: 'No usable scores across 3 trials (3 reported errors).',
        },
      },
    };

    beforeEach(() => {
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/${workspace}/agent-evaluate/jobs/${JOB_NAME}`,
          () => HttpResponse.json(failedJob)
        ),
        http.get(
          `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/${workspace}/agent-eval-results/${JOB_NAME}`,
          () =>
            HttpResponse.json({
              name: JOB_NAME,
              workspace,
              id: 'r1',
              job_id: JOB_NAME,
              bundle_ref: `${workspace}/bundle-fs#results/attempt-1`,
              created_at: '2026-08-01T00:05:00Z',
              updated_at: '2026-08-01T00:05:00Z',
              scores: {
                scores: [
                  { name: 'output-score.match', score_type: 'range', count: 0, nan_count: 3 },
                ],
              },
              summary: {
                task_count: 3,
                trial_count: 3,
                score_count: 3,
                error_count: 3,
                metric_coverage: {
                  'output-score': { match: { total: 3, scored: 0, failed: 3, missing: 0 } },
                },
              },
            })
        )
      );
    });

    it('explains the failure with the evaluator rollup and shows per-output coverage', async () => {
      renderDetail();
      expect(
        await screen.findByText('No usable scores across 3 trials (3 reported errors).')
      ).toBeInTheDocument();
      // The exit code no longer appears as the "Response": the rollup already explains the run.
      expect(screen.queryByText('Job exited with code 1')).not.toBeInTheDocument();
      expect(await screen.findByText('0/3 scored')).toBeInTheDocument();
      expect(screen.getByText('3 failed')).toBeInTheDocument();
      expect(screen.getByText('3 trials across 3 tasks; 3 trials reported errors.')).toBeInTheDocument();
    });
  });

  describe('Gym runner scores', () => {
    beforeEach(() => {
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/${workspace}/agent-evaluate/jobs/${JOB_NAME}`,
          () => HttpResponse.json(completedJob)
        ),
        http.get(
          `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/${workspace}/agent-eval-results/${JOB_NAME}`,
          () =>
            HttpResponse.json({
              name: JOB_NAME,
              workspace,
              id: 'r1',
              job_id: JOB_NAME,
              bundle_ref: `${workspace}/bundle-fs#results/attempt-1`,
              created_at: '2026-08-01T00:05:00Z',
              updated_at: '2026-08-01T00:05:00Z',
              scores: {
                scores: [
                  {
                    name: 'exact_match.exact_match',
                    score_type: 'range',
                    mean: 0.6,
                    count: 10,
                    nan_count: 0,
                  },
                  {
                    name: 'runner.gym.pass@1',
                    score_type: 'scalar',
                    value: 60.0,
                    nan_count: 0,
                  },
                ],
              },
            })
        )
      );
    });

    it('shows native and runner scores in separate sections', async () => {
      renderDetail();
      // Native score shown without a section heading
      expect(await screen.findByText('exact_match')).toBeInTheDocument();
      // Runner section heading appears
      expect(screen.getByText('Runner Scores')).toBeInTheDocument();
      // Runner score label shown (displayMetricName strips the leading "runner." segment)
      expect(screen.getByText('gym.pass@1')).toBeInTheDocument();
    });

    it('does not show the Runner metrics section when there are no runner scores', async () => {
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/${workspace}/agent-eval-results/${JOB_NAME}`,
          () =>
            HttpResponse.json({
              name: JOB_NAME,
              workspace,
              id: 'r1',
              job_id: JOB_NAME,
              bundle_ref: `${workspace}/bundle-fs#results/attempt-1`,
              created_at: '2026-08-01T00:05:00Z',
              updated_at: '2026-08-01T00:05:00Z',
              scores: {
                scores: [
                  {
                    name: 'exact_match.exact_match',
                    score_type: 'range',
                    mean: 0.6,
                    count: 10,
                    nan_count: 0,
                  },
                ],
              },
            })
        )
      );
      renderDetail();
      expect(await screen.findByText('exact_match')).toBeInTheDocument();
      expect(screen.queryByText('Runner Scores')).not.toBeInTheDocument();
    });
  });
});
