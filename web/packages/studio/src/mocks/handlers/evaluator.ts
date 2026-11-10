// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Revision, Task, Taskset } from '@nemo/sdk/generated/evals/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

const EVALS_URL = `${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/:workspace`;
const TASKSET_URL = `${EVALS_URL}/tasksets/:name`;

/** A stable 64-hex content hash per revision ordinal. */
const contentHash = (revision: number) => revision.toString(16).padStart(64, 'c');

/** Two tasksets in `default` and one in `staging`, so tests can prove workspace scoping. */
const mockTasksets: Taskset[] = [
  {
    id: 'taskset-sort-suite',
    name: 'sort-suite',
    workspace: 'default',
    description: 'Sorting algorithm tasks',
    tasks: ['default/sort-ints', 'default/sort-strings', 'default/sort-stable'],
    revision: 2,
    tags: { latest: 2, stable: 1 },
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-10-14T09:00:00Z',
  },
  {
    id: 'taskset-phishing-triage',
    name: 'phishing-triage',
    workspace: 'default',
    description: 'Email phishing classification',
    tasks: ['default/phish-1', 'default/phish-2'],
    revision: 1,
    tags: { latest: 1 },
    created_at: '2026-10-05T09:00:00Z',
    updated_at: '2026-10-05T09:00:00Z',
  },
  {
    id: 'taskset-staging-suite',
    name: 'staging-suite',
    workspace: 'staging',
    tasks: ['staging/smoke'],
    revision: 1,
    tags: { latest: 1 },
    created_at: '2026-10-07T09:00:00Z',
    updated_at: '2026-10-07T09:00:00Z',
  },
];

const HARBOR_HASH = 'b7c2'.repeat(16);

/** The `sort-suite` members: two Harbor tasks and one inline task. `sort-stable` is left out so
 *  a taskset can reference a task that no longer resolves. */
const mockTasks: Task[] = [
  {
    id: 'task-sort-ints',
    name: 'sort-ints',
    workspace: 'default',
    revision: 1,
    tags: { latest: 1 },
    metadata: [{ key: 'difficulty', value: 'easy' }],
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-10-01T09:00:00Z',
    spec: {
      kind: 'harbor',
      native_task_id: 'sort-ints',
      source: {
        kind: 'fileset-archive',
        fileset_ref: 'default/sort-suite-files#tasks/sort-ints.tar.gz',
        files_hash: HARBOR_HASH,
      },
      harbor_hash: { digest: HARBOR_HASH, harbor_version: '0.1.0' },
      instruction: 'Sort the integers in input.txt and write them to output.txt.',
      config: {
        environment: { docker_image: 'python:3.12-slim', cpus: 1, memory_mb: 2048 },
        verifier: { timeout_sec: 120 },
      },
    },
  },
  {
    id: 'task-sort-strings',
    name: 'sort-strings',
    workspace: 'default',
    revision: 2,
    tags: { latest: 2 },
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-10-02T09:00:00Z',
    spec: {
      kind: 'evaluator',
      intent: 'Sort a list of strings case-insensitively',
      inputs: { instruction: 'Sort: banana, Apple, cherry' },
      reference: { expected: ['Apple', 'banana', 'cherry'] },
    },
  },
];

/** Name filter as the SDK serializes it with `qs` brackets: exact, `$in`, or `%text%` `$like`. */
const nameMatcher = (request: Request) => {
  const params = new URL(request.url).searchParams;
  const exact = params.get('filter[name]');
  const names = params.getAll('filter[name][$in]');
  const like = params.get('filter[name][$like]')?.replaceAll('%', '').toLowerCase();
  return (name: string) =>
    (!exact || name === exact) &&
    (!names.length || names.includes(name)) &&
    (!like || name.toLowerCase().includes(like));
};

const notFound = (what: string) =>
  HttpResponse.json({ detail: `${what} not found` }, { status: 404 });

const paginate = <T>(request: Request, rows: T[]) => {
  const url = new URL(request.url);
  const page = Number(url.searchParams.get('page') ?? 1);
  const pageSize = Number(url.searchParams.get('page_size') ?? 50);
  const data = rows.slice((page - 1) * pageSize, page * pageSize);
  return HttpResponse.json({
    data,
    pagination: {
      page,
      page_size: pageSize,
      current_page_size: data.length,
      total_pages: Math.max(1, Math.ceil(rows.length / pageSize)),
      total_results: rows.length,
    },
  });
};

const findTaskset = (workspace: unknown, name: unknown) =>
  mockTasksets.find((t) => t.workspace === workspace && t.name === name);

export const evaluatorHandlers = [
  // nemo-evals agent-evaluate jobs — Studio's agent-evaluation feature runs
  // here. Default to an empty list / 404 so the list + detail routes render
  // their empty/not-found states in tests without per-test overrides.
  http.get(`${EVALS_URL}/agent-evaluate/jobs`, () =>
    HttpResponse.json({ data: [], pagination: { total: 0, page: 1, page_size: 50 } })
  ),
  http.get(`${EVALS_URL}/agent-evaluate/jobs/:name`, () => notFound('Job')),
  http.get(`${EVALS_URL}/agent-evaluate/jobs/:name/status`, () =>
    HttpResponse.json({ name: '', status: 'unknown' })
  ),
  http.get(`${EVALS_URL}/agent-eval-results/:name`, () => notFound('Result')),

  http.get(`${EVALS_URL}/tasks`, ({ request, params }) => {
    const matches = nameMatcher(request);
    return paginate(
      request,
      mockTasks.filter((t) => t.workspace === params.workspace && matches(t.name))
    );
  }),

  // Tasksets — read-only, as the Studio library is.
  http.get(`${EVALS_URL}/tasksets`, ({ request, params }) => {
    const matches = nameMatcher(request);
    return paginate(
      request,
      mockTasksets.filter((t) => t.workspace === params.workspace && matches(t.name))
    );
  }),
  http.get(TASKSET_URL, ({ params }) => {
    const taskset = findTaskset(params.workspace, params.name);
    return taskset ? HttpResponse.json(taskset) : notFound('Taskset');
  }),
  http.get(`${TASKSET_URL}/revisions`, ({ request, params }) => {
    const taskset = findTaskset(params.workspace, params.name);
    if (!taskset) return notFound('Taskset');
    // Newest first, as the server orders them.
    const revisions: Revision[] = Array.from({ length: taskset.revision }, (_, i) => {
      const revision = taskset.revision - i;
      return {
        revision,
        content_hash: contentHash(revision),
        tags: Object.keys(taskset.tags ?? {}).filter((tag) => taskset.tags?.[tag] === revision),
        created_at: taskset.created_at,
      };
    });
    return paginate(request, revisions);
  }),
];
