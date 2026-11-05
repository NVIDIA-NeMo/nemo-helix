// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import { makeEvaluatorJob } from '@studio/mocks/evaluation/jobs';
import { server } from '@studio/mocks/node';
import { useAgentDetails } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const OTHER_WORKSPACE = 'other-ws';
const agentName = 'my-agent';
const evaluationsUrl = '*/apis/intake/v2/workspaces/:workspace/evaluations';

const evaluation = (name: string): EvaluationResponse => ({
  id: `id-${name}`,
  name,
  workspace: DEFAULT_WORKSPACE,
  experiment_ids: [],
  experiment_group_id: '',
  dataset_name: 'dataset',
});

it('resolves older jobs beyond a five-job preview and discovers missing Intake rows by publication', async () => {
  const requestedPages: number[] = [];
  const evaluationPageSizes: number[] = [];
  const jobs = Array.from({ length: 50 }, (_, index) =>
    makeEvaluatorJob({
      id: `job-${index}`,
      name: `job-${index}`,
      spec: {
        target:
          index === 0
            ? { kind: 'agent', agent: { name: agentName } }
            : index < 6
              ? { kind: 'fabric', source: { agent: `${DEFAULT_WORKSPACE}/${agentName}` } }
              : index === 6
                ? { kind: 'gym', source: { agent: `${OTHER_WORKSPACE}/${agentName}` } }
                : { kind: 'model', name: agentName },
        publication: { intake: { evaluation_id: `eval-${index}` } },
      },
    })
  );
  const olderJob = makeEvaluatorJob({
    id: 'older-job',
    name: 'older-job',
    spec: {
      target: { kind: 'harbor', source: { agent: `${DEFAULT_WORKSPACE}/${agentName}` } },
      publication: { intake: { evaluation_id: 'older-eval' } },
    },
  });
  server.use(
    http.get('*/apis/jobs/v2/workspaces/:workspace/jobs', ({ request }) => {
      const page = Number(new URL(request.url).searchParams.get('page'));
      requestedPages.push(page);
      return HttpResponse.json({ data: page === 1 ? jobs : [olderJob] });
    }),
    http.get(evaluationsUrl, ({ request }) => {
      evaluationPageSizes.push(Number(new URL(request.url).searchParams.get('page_size')));
      return HttpResponse.json({
        data: Array.from({ length: 6 }, (_, index) => evaluation(`eval-${index}`)),
      });
    }),
    http.get(`${evaluationsUrl}/:name`, ({ params }) => {
      expect(params.name).toBe('older-eval');
      return HttpResponse.json(evaluation('older-eval'));
    })
  );

  const { result } = renderHook(
    () => useAgentDetails({ workspace: DEFAULT_WORKSPACE, agentName }),
    {
      wrapper: TestProviders,
    }
  );

  await waitFor(() => expect(result.current.agentEvals).toHaveLength(7));
  expect(requestedPages).toContain(2);
  expect(evaluationPageSizes.every((size) => size > 5)).toBe(true);
  expect(result.current.agentJobs).toHaveLength(7);
  expect(result.current.agentJobs.at(-1)).toMatchObject({
    name: 'older-job',
    evaluationName: 'older-eval',
    status: 'completed',
  });
  expect(result.current.agentEvals.at(-1)?.name).toBe('older-eval');
});
