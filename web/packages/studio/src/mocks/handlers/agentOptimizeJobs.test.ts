// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';

const listOptimizeJobs = (filter?: unknown, workspace = 'default') => {
  const url = new URL(
    `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/${workspace}/jobs/optimize`,
    window.location.origin
  );
  if (filter !== undefined) url.searchParams.set('filter', JSON.stringify(filter));
  return fetch(url);
};

describe('mock optimize jobs handler', () => {
  it('returns every job when no filter is sent', async () => {
    const response = await listOptimizeJobs();
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({ pagination: { total_results: 3 } });
  });

  it('scopes the list to the requested workspace', async () => {
    const response = await listOptimizeJobs(undefined, 'staging');
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.data.map((job: { name: string }) => job.name)).toEqual(['staging-sweep-1']);
    expect(body.pagination.total_results).toBe(1);
  });

  it('applies a supported spec.agent filter', async () => {
    const response = await listOptimizeJobs({
      'spec.agent': { $in: ['react-agent', 'default/react-agent'] },
    });
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.data.map((job: { name: string }) => job.name)).toEqual([
      'brevity-sweep-3',
      'accuracy-sweep-1',
    ]);
  });

  it('applies a $like name search with wildcards', async () => {
    const response = await listOptimizeJobs({ name: { $like: '%BREVITY%' } });
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.data.map((job: { name: string }) => job.name)).toEqual(['brevity-sweep-3']);
  });

  it.each([
    ['a nested spec object', { spec: { agent: { $in: ['react-agent'] } } }],
    ['a field with no operator', { name: 'brevity-sweep-3' }],
    ['an unrecognized field', { status: { $eq: 'completed' } }],
    ['an empty object', {}],
    ['a non-object filter', ['react-agent']],
  ])('rejects %s instead of matching every job', async (_label, filter) => {
    const response = await listOptimizeJobs(filter);
    expect(response.status).toBe(400);
  });

  it('rejects a filter that is not valid JSON', async () => {
    const url = new URL(
      `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/default/jobs/optimize`,
      window.location.origin
    );
    url.searchParams.set('filter', '{not json');
    expect((await fetch(url)).status).toBe(400);
  });
});
