// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { OptimizeJob } from '@nemo/sdk/generated/agents/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

const OPTIMIZE_JOBS_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/optimize`;

/**
 * Two studies for `react-agent` and one for another agent, so a test can prove the request scopes
 * the list to one agent — the handler below filters exactly as the server does. One study lives in
 * another workspace, so a test can prove the list is scoped to the requested workspace too.
 */
export const mockOptimizeJobs: OptimizeJob[] = [
  {
    id: 'opt-8f21',
    name: 'brevity-sweep-3',
    workspace: 'default',
    status: 'completed',
    created_at: '2026-08-14T09:00:00Z',
    updated_at: '2026-08-14T11:00:00Z',
    spec: { optimize_config: 'optimize-brevity.yaml', agent: 'react-agent' },
  },
  {
    id: 'opt-9a03',
    name: 'accuracy-sweep-1',
    workspace: 'default',
    status: 'active',
    created_at: '2026-08-13T09:00:00Z',
    updated_at: '2026-08-13T09:30:00Z',
    // Workspace-qualified reference for the same agent — must still match.
    spec: { optimize_config: 'optimize-accuracy.yaml', agent: 'default/react-agent' },
  },
  {
    id: 'opt-4400',
    name: 'other-agent-sweep',
    workspace: 'default',
    status: 'error',
    created_at: '2026-08-12T09:00:00Z',
    spec: { optimize_config: 'optimize-other.yaml', agent: 'other-agent' },
  },
  {
    id: 'opt-7c15',
    name: 'staging-sweep-1',
    workspace: 'staging',
    status: 'completed',
    created_at: '2026-08-11T09:00:00Z',
    spec: { optimize_config: 'optimize-brevity.yaml', agent: 'react-agent' },
  },
];

interface FilterQuery {
  'spec.agent'?: { $in: string[] };
  name?: { $like: string };
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isAgentClause = (value: unknown): value is { $in: string[] } =>
  isRecord(value) &&
  Object.keys(value).length === 1 &&
  Array.isArray(value.$in) &&
  value.$in.every((entry) => typeof entry === 'string');

const isNameClause = (value: unknown): value is { $like: string } =>
  isRecord(value) && Object.keys(value).length === 1 && typeof value.$like === 'string';

/**
 * The slice of the platform filter syntax this endpoint is called with: exact `$in` on the dotted
 * spec path, `$like` with `%` wildcards on the name. Anything else — a nested `spec` object, a
 * missing operator, an unrecognized field — is rejected the way the server rejects it, so a
 * malformed filter fails the test instead of quietly matching every row. Returns `null` when the
 * filter is not one this endpoint supports.
 */
const parseFilter = (raw: string): FilterQuery | null => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  const entries = Object.entries(parsed);
  if (entries.length === 0) return null;

  const filter: FilterQuery = {};
  for (const [key, value] of entries) {
    if (key === 'spec.agent' && isAgentClause(value)) filter['spec.agent'] = value;
    else if (key === 'name' && isNameClause(value)) filter.name = value;
    else return null;
  }
  return filter;
};

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** `%` is the platform's wildcard; a substring search arrives as `%text%`. */
const likeMatches = (value: string, pattern: string): boolean =>
  new RegExp(`^${pattern.split('%').map(escapeRegExp).join('.*')}$`, 'i').test(value);

const applyFilter = (jobs: OptimizeJob[], filter: FilterQuery): OptimizeJob[] => {
  const agents = filter['spec.agent']?.$in;
  const like = filter.name?.$like;
  return jobs.filter((job) => {
    if (agents && !agents.includes(String(job.spec?.agent ?? ''))) return false;
    return like === undefined || likeMatches(job.name, like);
  });
};

export const agentOptimizeJobsHandlers = [
  http.get(OPTIMIZE_JOBS_URL, ({ request, params }) => {
    const url = new URL(request.url);
    const workspace = String(params.workspace);
    const page = Number(url.searchParams.get('page') ?? 1);
    const pageSize = Number(url.searchParams.get('page_size') ?? 50);
    const raw = url.searchParams.get('filter');
    const filter = raw ? parseFilter(raw) : {};
    if (!filter) {
      return HttpResponse.json({ detail: `Unsupported filter: ${raw}` }, { status: 400 });
    }
    const matches = applyFilter(
      mockOptimizeJobs.filter((job) => job.workspace === workspace),
      filter
    );
    const data = matches.slice((page - 1) * pageSize, page * pageSize);
    return HttpResponse.json({
      data,
      pagination: {
        page,
        page_size: pageSize,
        current_page_size: data.length,
        total_pages: Math.max(1, Math.ceil(matches.length / pageSize)),
        total_results: matches.length,
      },
    });
  }),
];
