// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { fetchEvaluatorJobs } from '@studio/api/evaluation/evaluator-jobs';
import { makeEvaluatorJob } from '@studio/mocks/evaluation/jobs';
import { server } from '@studio/mocks/node';
import { http, HttpResponse } from 'msw';

it('scopes every job page to both evaluator sources and local registered-agent target forms', async () => {
  const filters: unknown[] = [];
  const pages: number[] = [];
  server.use(
    http.get('*/apis/jobs/v2/workspaces/:workspace/jobs', ({ request }) => {
      const params = new URL(request.url).searchParams;
      filters.push(JSON.parse(params.get('filter') ?? '{}'));
      const page = Number(params.get('page'));
      pages.push(page);
      return HttpResponse.json({
        data: page === 1 ? Array.from({ length: 50 }, () => makeEvaluatorJob()) : [],
      });
    })
  );

  await fetchEvaluatorJobs(DEFAULT_WORKSPACE, new AbortController().signal, undefined, 'my-agent');

  expect(pages).toEqual([1, 2]);
  for (const filter of filters) {
    expect(filter).toEqual({
      $and: [
        { source: { $in: ['nemo-evals', 'nemo-evals.agent-evaluate'] } },
        {
          $or: [
            {
              'spec.target.kind': { $eq: 'agent' },
              'spec.target.agent.name': { $in: ['my-agent', `${DEFAULT_WORKSPACE}/my-agent`] },
            },
            {
              'spec.target.kind': { $in: ['fabric', 'harbor', 'gym'] },
              'spec.target.source.agent': { $in: ['my-agent', `${DEFAULT_WORKSPACE}/my-agent`] },
            },
            {
              'spec.target.format': { $eq: 'generic' },
              'spec.target.name': { $in: ['my-agent', `${DEFAULT_WORKSPACE}/my-agent`] },
            },
          ],
        },
      ],
    });
  }
});
