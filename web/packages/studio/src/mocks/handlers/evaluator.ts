// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Revision, Task, Taskset, TasksetInput } from '@nemo/sdk/generated/evals/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

const EVALS_URL = `${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/:workspace`;
const TASKSET_URL = `${EVALS_URL}/tasksets/:name`;

/** A stable 64-hex content hash per revision ordinal. */
const contentHash = (revision: number) => revision.toString(16).padStart(64, 'c');

/** Two tasksets in `default` and one in `staging`, so tests can prove workspace scoping. Both
 *  `default` tasksets pin `sort-strings`, as the server stores members: `sort-suite` to its first
 *  revision (behind latest) and `phishing-triage` to its latest. */
const mockTasksets: Taskset[] = [
  {
    id: 'taskset-sort-suite',
    name: 'sort-suite',
    workspace: 'default',
    description: 'Sorting algorithm tasks',
    tasks: ['default/sort-ints', `default/sort-strings#${contentHash(1)}`, 'default/sort-stable'],
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
    tasks: ['default/phish-1', 'default/phish-2', `default/sort-strings#${contentHash(2)}`],
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

/** A text filter on `field` as the SDK serializes it with `qs` brackets: exact, `$in`, or
 *  `%text%` `$like`. */
const textMatcher = (params: URLSearchParams, field: string) => {
  const exact = params.get(`filter[${field}]`);
  const values = params.getAll(`filter[${field}][$in]`);
  const like = params.get(`filter[${field}][$like]`)?.replaceAll('%', '').toLowerCase();
  return (value = '') =>
    (!exact || value === exact) &&
    (!values.length || values.includes(value)) &&
    (!like || value.toLowerCase().includes(like));
};

/** List filters shared by tasks and tasksets: name, description, kind, a tag, an `updated_at`
 *  range, and a member task (`workspace/name` at any revision). */
const recordMatcher = (request: Request) => {
  const params = new URL(request.url).searchParams;
  const name = textMatcher(params, 'name');
  const description = textMatcher(params, 'description');
  const kind = textMatcher(params, 'kind');
  const tag = params.get('filter[tags]');
  const member = params.get('filter[tasks]');
  const from = params.get('filter[updated_at][$gte]');
  const to = params.get('filter[updated_at][$lte]');
  return (r: Partial<Task & Taskset> & Pick<Task, 'name' | 'updated_at'>) =>
    name(r.name) &&
    description(r.description) &&
    kind(r.spec?.kind) &&
    (!tag || tag in (r.tags ?? {})) &&
    (!member || (r.tasks ?? []).some((t) => t.split('#')[0] === member)) &&
    (!from || Date.parse(r.updated_at) >= Date.parse(from)) &&
    (!to || Date.parse(r.updated_at) <= Date.parse(to));
};

/** Newest first, as the server orders them; `tags` holds the tags pointing at each. */
const revisionsOf = (record: Task | Taskset): Revision[] =>
  Array.from({ length: record.revision }, (_, i) => {
    const revision = record.revision - i;
    return {
      revision,
      content_hash: contentHash(revision),
      tags: Object.keys(record.tags ?? {}).filter((tag) => record.tags?.[tag] === revision),
      created_at: record.created_at,
    };
  });

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
const findTask = (workspace: unknown, name: unknown) =>
  mockTasks.find((t) => t.workspace === workspace && t.name === name);

/** Pins a member ref to its task's latest content hash, as the server does on write;
 *  undefined when the task doesn't exist. */
const resolveTaskRef = (workspace: string, ref: string): string | undefined => {
  const [path, version] = ref.split('#');
  const [ws, name] = path.includes('/') ? path.split('/') : [workspace, path];
  const task = findTask(ws, name);
  return task && `${ws}/${name}#${version ?? contentHash(task.revision)}`;
};

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

  // Tasks and tasksets.
  http.get(`${EVALS_URL}/tasks`, ({ request, params }) => {
    const matches = recordMatcher(request);
    return paginate(
      request,
      mockTasks.filter((t) => t.workspace === params.workspace && matches(t))
    );
  }),
  http.get(`${EVALS_URL}/tasks/:name`, ({ params }) => {
    const task = findTask(params.workspace, params.name);
    return task ? HttpResponse.json(task) : notFound('Task');
  }),
  http.get(`${EVALS_URL}/tasks/:name/revisions`, ({ request, params }) => {
    const task = findTask(params.workspace, params.name);
    return task ? paginate(request, revisionsOf(task)) : notFound('Task');
  }),

  http.get(`${EVALS_URL}/tasksets`, ({ request, params }) => {
    const matches = recordMatcher(request);
    return paginate(
      request,
      mockTasksets.filter((t) => t.workspace === params.workspace && matches(t))
    );
  }),
  http.post(TASKSET_URL, async ({ request, params }) => {
    const workspace = String(params.workspace);
    const name = String(params.name);
    if (findTaskset(workspace, name)) {
      return HttpResponse.json({ detail: `Taskset ${name} already exists` }, { status: 409 });
    }
    const { tasks = [], description, files_ref, metadata } = (await request.json()) as TasksetInput;
    const resolved = tasks.map((ref) => resolveTaskRef(workspace, ref));
    if (resolved.some((ref) => !ref)) {
      return HttpResponse.json({ detail: 'Unknown task reference' }, { status: 422 });
    }
    const now = new Date().toISOString();
    const taskset: Taskset = {
      id: `taskset-${name}`,
      name,
      workspace,
      description,
      files_ref,
      metadata,
      tasks: resolved as string[],
      revision: 1,
      tags: { latest: 1 },
      created_at: now,
      updated_at: now,
    };
    mockTasksets.push(taskset);
    return HttpResponse.json(taskset, { status: 201 });
  }),
  http.get(TASKSET_URL, ({ params }) => {
    const taskset = findTaskset(params.workspace, params.name);
    return taskset ? HttpResponse.json(taskset) : notFound('Taskset');
  }),
  http.get(`${TASKSET_URL}/revisions`, ({ request, params }) => {
    const taskset = findTaskset(params.workspace, params.name);
    return taskset ? paginate(request, revisionsOf(taskset)) : notFound('Taskset');
  }),
];
