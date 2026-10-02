// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getEvaluatorGetEvalResultQueryKey } from '@nemo/sdk/generated/evaluator/evaluator-plugin-eval-results-routes';
import {
  getEvaluatorGetEvaluateJobQueryKey,
  getEvaluatorGetEvaluateJobResultQueryKey,
} from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { EvaluationResultDetailsRoute } from '@studio/routes/evaluation/EvaluationResultDetailsRoute';
import { getEvaluationResultDetailsRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';
import { delay, http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const JOB_NAME = 'dataset-eval-run';

const completedJob = {
  name: JOB_NAME,
  workspace,
  status: 'completed',
  created_at: '2026-08-01T00:00:00Z',
  spec: { target: { name: 'my-model' }, dataset: 'my-dataset#data.jsonl' },
};

const jobUrl = mockApiUrl(getEvaluatorGetEvaluateJobQueryKey, ':workspace', ':name');
const scoresUrl = mockApiUrl(getEvaluatorGetEvalResultQueryKey, ':workspace', ':name');
const rowsMetadataUrl = mockApiUrl(
  getEvaluatorGetEvaluateJobResultQueryKey,
  ':workspace',
  ':job',
  'row-scores'
);

const renderDetail = () =>
  renderRoute(<EvaluationResultDetailsRoute />, {
    history: getEvaluationResultDetailsRoute(workspace, JOB_NAME),
    routes: [
      {
        path: ROUTES.workspace.evaluationResultDetails,
        element: <EvaluationResultDetailsRoute />,
      },
    ],
  });

const expectNoEmptyResults = () => {
  expect(screen.queryByText('No scores recorded for this evaluation.')).not.toBeInTheDocument();
  expect(
    screen.queryByText('No per-row results recorded for this evaluation.')
  ).not.toBeInTheDocument();
};

describe('EvaluationResultDetailsRoute', () => {
  it('shows a loading state, not empty or failed content, while the job is fetching', async () => {
    server.use(
      http.get(jobUrl, async () => {
        await delay('infinite');
        return HttpResponse.json(completedJob);
      })
    );

    renderDetail();

    expect(await screen.findByTestId('spinner')).toBeInTheDocument();
    expect(screen.queryByText('Failed to Load Job')).not.toBeInTheDocument();
    expectNoEmptyResults();
  });

  it('shows result spinners, not empty content, while a completed job’s results are fetching', async () => {
    server.use(
      http.get(jobUrl, () => HttpResponse.json(completedJob)),
      http.get(scoresUrl, async () => {
        await delay('infinite');
        return HttpResponse.json({});
      }),
      http.get(rowsMetadataUrl, async () => {
        await delay('infinite');
        return HttpResponse.json({});
      })
    );

    renderDetail();

    expect(await screen.findByLabelText('Loading scores...')).toBeInTheDocument();
    expect(screen.getByLabelText('Loading row results...')).toBeInTheDocument();
    expectNoEmptyResults();
  });

  it.each(['created', 'paused'])(
    'shows the pending message, not empty content, while the job is %s',
    async (status) => {
      server.use(http.get(jobUrl, () => HttpResponse.json({ ...completedJob, status })));

      renderDetail();

      expect(
        await screen.findByText('Scores are computed once the job reaches a terminal state.')
      ).toBeInTheDocument();
      expect(
        screen.getByText('Row results are computed once the job reaches a terminal state.')
      ).toBeInTheDocument();
      expectNoEmptyResults();
    }
  );

  it('shows only the load error when the job cannot be fetched', async () => {
    server.use(http.get(jobUrl, () => HttpResponse.json({ detail: 'boom' }, { status: 500 })));

    renderDetail();

    expect(await screen.findByText('Failed to Load Job')).toBeInTheDocument();
    expectNoEmptyResults();
  });

  it('renders aggregate scores once a completed job’s results load', async () => {
    server.use(
      http.get(jobUrl, () => HttpResponse.json(completedJob)),
      http.get(scoresUrl, () =>
        HttpResponse.json({
          scores: { scores: [{ name: 'exact-match', score_type: 'scalar', value: 0.75 }] },
        })
      ),
      http.get(rowsMetadataUrl, () => HttpResponse.json({}))
    );

    renderDetail();

    expect(await screen.findByText('exact-match')).toBeInTheDocument();
    expect(screen.queryByLabelText('Loading row results...')).not.toBeInTheDocument();
  });
});
