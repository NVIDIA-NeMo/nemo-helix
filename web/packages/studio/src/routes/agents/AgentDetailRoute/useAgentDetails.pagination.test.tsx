// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_LONG, JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { EvaluationResponse, HelixJobResponse } from '@nemo/sdk/generated/platform/schema';
import { makeEvaluatorJob } from '@studio/mocks/evaluation/jobs';
import { server } from '@studio/mocks/node';
import { AGENT_EVALUATIONS_PAGE_SIZE } from '@studio/routes/agents/AgentDetailRoute/constants';
import { useAgentDetails } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { act, renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const OTHER_WORKSPACE = 'other-ws';
const agentName = 'my-agent';
const evaluationsUrl = '*/apis/intake/v2/workspaces/:workspace/evaluations';

afterEach(() => vi.useRealTimers());

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

it('bounds publication hydration to 100 recent jobs but links displayed evaluations across all API pages', async () => {
  const requestedPages: number[] = [];
  const lookups: string[] = [];
  const jobs = Array.from({ length: 151 }, (_, index) =>
    makeEvaluatorJob({
      id: `job-${index}`,
      name: `job-${index}`,
      spec: {
        target: { kind: 'fabric', source: { agent: agentName } },
        publication: { intake: { evaluation_id: `eval-${index}` } },
      },
    })
  );
  server.use(
    http.get('*/apis/jobs/v2/workspaces/:workspace/jobs', ({ request }) => {
      const page = Number(new URL(request.url).searchParams.get('page'));
      requestedPages.push(page);
      return HttpResponse.json({ data: jobs.slice((page - 1) * 50, page * 50) });
    }),
    http.get(evaluationsUrl, () => HttpResponse.json({ data: [evaluation('eval-150')] })),
    http.get(`${evaluationsUrl}/:name`, ({ params }) => {
      lookups.push(String(params.name));
      return HttpResponse.json(evaluation(String(params.name)));
    })
  );
  const { result } = renderHook(
    () => useAgentDetails({ workspace: DEFAULT_WORKSPACE, agentName }),
    { wrapper: TestProviders }
  );
  await waitFor(() => expect(result.current.agentEvals).toHaveLength(AGENT_EVALUATIONS_PAGE_SIZE));
  expect(requestedPages).toEqual([1, 2, 3, 4]);
  expect(lookups).toHaveLength(AGENT_EVALUATIONS_PAGE_SIZE);
  expect(lookups).not.toContain('eval-100');
  expect(result.current.agentEvals.some((row) => row.name === 'eval-150')).toBe(true);
  expect(result.current.agentJobs.find((row) => row.evaluationName === 'eval-150')?.name).toBe(
    'job-150'
  );
});

it('ignores unrelated live jobs and retries only unpublished evaluations until discovery', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  let jobRequests = 0;
  let published = false;
  const lookups: string[] = [];
  const jobs: HelixJobResponse[] = [
    makeEvaluatorJob({
      id: 'unrelated',
      status: 'active',
      spec: { target: { kind: 'fabric', source: { agent: 'another-agent' } } },
    }),
    ...['ready', 'delayed'].map((name) =>
      makeEvaluatorJob({
        id: name,
        spec: {
          target: { kind: 'fabric', source: { agent: agentName } },
          publication: { intake: { evaluation_id: name } },
        },
      })
    ),
  ];
  server.use(
    http.get('*/apis/jobs/v2/workspaces/:workspace/jobs', () => {
      jobRequests++;
      return HttpResponse.json({ data: jobs });
    }),
    http.get(evaluationsUrl, () => HttpResponse.json({ data: [] })),
    http.get(`${evaluationsUrl}/:name`, ({ params }) => {
      const name = String(params.name);
      lookups.push(name);
      return name === 'delayed' && !published
        ? new HttpResponse(null, { status: 404 })
        : HttpResponse.json(evaluation(name));
    })
  );
  const { result } = renderHook(
    () => useAgentDetails({ workspace: DEFAULT_WORKSPACE, agentName }),
    { wrapper: TestProviders }
  );
  await waitFor(() => expect(result.current.agentEvals).toHaveLength(1));
  await waitFor(() => expect(lookups).toHaveLength(2));
  await act(() => vi.advanceTimersByTimeAsync(JOB_POLLING_INTERVAL_MS + 1));
  expect(jobRequests).toBe(1);
  expect(lookups).toHaveLength(2);
  published = true;
  await act(() => vi.advanceTimersByTimeAsync(JOB_POLLING_INTERVAL_LONG));
  vi.useRealTimers();
  await waitFor(() => expect(result.current.agentEvals).toHaveLength(2));
  expect(lookups.filter((name) => name === 'ready')).toHaveLength(1);
  expect(lookups.filter((name) => name === 'delayed')).toHaveLength(2);
  expect(jobRequests).toBe(1);
});
