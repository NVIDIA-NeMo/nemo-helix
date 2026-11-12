// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';

const evals = async (path: string) =>
  fetch(new URL(`${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/${path}`, window.location.origin));

const names = async (path: string) =>
  (await (await evals(path)).json()).data.map((t: { name: string }) => t.name);

describe('mock taskset handlers', () => {
  it('lists tasksets scoped to the workspace', async () => {
    expect(await names('staging/tasksets')).toEqual(['staging-suite']);
  });

  it('filters tasksets by name substring', async () => {
    expect(await names('default/tasksets?filter[name][$like]=%25PHISH%25')).toEqual([
      'phishing-triage',
    ]);
  });

  it.each([
    ['description', 'filter[description][$like]=%25sorting%25', ['sort-suite']],
    ['tag', 'filter[tags]=stable', ['sort-suite']],
    ['updated range', 'filter[updated_at][$gte]=2026-10-10T00:00:00Z', ['sort-suite']],
    ['updated upper bound', 'filter[updated_at][$lte]=2026-10-10T00:00:00Z', ['phishing-triage']],
  ])('filters tasksets by %s', async (_, query, expected) => {
    expect(await names(`default/tasksets?${query}`)).toEqual(expected);
  });

  it('filters tasks by a name list', async () => {
    expect(
      await names('default/tasks?filter[name][$in]=sort-ints&filter[name][$in]=missing')
    ).toEqual(['sort-ints']);
  });

  it('filters tasks by kind and tasksets by member task at any revision', async () => {
    expect(await names('default/tasks?filter[kind][$in]=evaluator')).toEqual(['sort-strings']);
    expect(await names('default/tasksets?filter[tasks]=default/sort-strings')).toEqual([
      'sort-suite',
      'phishing-triage',
    ]);
  });

  it('gets a task with its revisions and 404s on an unknown one', async () => {
    await expect((await evals('default/tasks/sort-strings')).json()).resolves.toMatchObject({
      revision: 2,
    });
    const page = await (await evals('default/tasks/sort-strings/revisions')).json();
    expect(page.data.map((r: { revision: number }) => r.revision)).toEqual([2, 1]);
    expect((await evals('default/tasks/missing')).status).toBe(404);
  });

  it('gets a taskset and 404s on an unknown one', async () => {
    await expect((await evals('default/tasksets/sort-suite')).json()).resolves.toMatchObject({
      revision: 2,
    });
    expect((await evals('default/tasksets/missing')).status).toBe(404);
  });

  it('lists revisions newest first with their tags', async () => {
    const page = await (await evals('default/tasksets/sort-suite/revisions')).json();
    expect(
      page.data.map((r: { revision: number; tags: string[] }) => [r.revision, r.tags])
    ).toEqual([
      [2, ['latest']],
      [1, ['stable']],
    ]);
  });
});
