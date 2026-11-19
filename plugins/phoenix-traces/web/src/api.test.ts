// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { listProjects, listSpans, type ApiHost } from './api';
import { useProjects, useSpans } from './queries';
import type { PhoenixSpan } from './traces';

vi.mock('@tanstack/react-query', () => ({ useQuery: (options: unknown) => options }));

const ORIGIN = 'https://platform.invalid';
const WORKSPACE = 'alpha-workspace';
const BASE = `${ORIGIN}/apis/plugins/phoenix/v2/workspaces/${WORKSPACE}`;
const server = setupServer();
const makeHost = (): ApiHost => ({
  apiBaseUrl: `${ORIGIN}/`,
  workspaceId: WORKSPACE,
  auth: { accessToken: '', getAccessToken: vi.fn(() => 'platform-access-token') },
});
const span: PhoenixSpan = {
  name: 'persisted call',
  context: { trace_id: 'trace-a', span_id: 'child-a' },
  parent_id: 'root-a',
  span_kind: 'LLM',
  start_time: '2026-11-18T12:00:00Z',
  end_time: '2026-11-18T12:00:01Z',
  status_code: 'OK',
  status_message: '',
  attributes: { 'llm.token_count.total': 12, 'provider.raw': { nested: [null, '雪'] } },
};

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => {
  server.resetHandlers();
  vi.restoreAllMocks();
});
afterAll(() => server.close());

it('pages projects in the URL workspace and obtains the current token per request', async () => {
  const host = makeHost();
  vi.mocked(host.auth.getAccessToken)
    .mockReturnValueOnce('first-access-token')
    .mockReturnValueOnce('refreshed-access-token');
  const authorization: (string | null)[] = [];
  server.use(http.get(`${BASE}/projects`, ({ request }) => {
    authorization.push(request.headers.get('Authorization'));
    const cursor = new URL(request.url).searchParams.get('cursor');
    return HttpResponse.json({
      data: [{ name: cursor ? 'project-2' : 'project-1' }],
      next_cursor: cursor ? null : 'next-page',
    });
  }));
  expect(await listProjects(host)).toEqual(['project-1', 'project-2']);
  expect(authorization).toEqual(['Bearer first-access-token', 'Bearer refreshed-access-token']);
});

it('queries arbitrary project names without putting them in an upstream path', async () => {
  const project = 'real project / 雪 ? #';
  server.use(http.get(`${BASE}/spans`, ({ request }) => {
    expect(new URL(request.url).searchParams.get('project')).toBe(project);
    return HttpResponse.json({ data: [span], next_cursor: null });
  }));
  const [result] = await listSpans(makeHost(), project);
  expect(result).toMatchObject({ id: 'child-a', traceId: 'trace-a', parentId: 'root-a', status: 'OK' });
  expect(result.attributes).toEqual(span.attributes);
  expect(result.end - result.start).toBe(1000);
});

it('keeps an empty project filter for unassigned Intake spans and handles running spans', async () => {
  const observed = Date.parse('2026-11-18T12:00:05Z');
  vi.spyOn(Date, 'now').mockReturnValue(observed);
  server.use(http.get(`${BASE}/spans`, ({ request }) => {
    expect(new URL(request.url).searchParams.get('project')).toBe('');
    return HttpResponse.json({ data: [{ ...span, end_time: null }], next_cursor: null });
  }));
  expect((await listSpans(makeHost(), ''))[0].end).toBe(observed);
});

it('does not return misleading partial traces when the page cap is exceeded', async () => {
  let requests = 0;
  server.use(http.get(`${BASE}/spans`, () => {
    requests++;
    return HttpResponse.json({ data: [span], next_cursor: `page-${requests}` });
  }));
  await expect(listSpans(makeHost(), 'demo')).rejects.toThrow('pagination limit');
  expect(requests).toBe(10);
});

it('surfaces Intake or gateway failures rather than inventing sample data', async () => {
  server.use(http.get(`${BASE}/projects`, () => new HttpResponse(null, { status: 503 })));
  await expect(listProjects(makeHost())).rejects.toThrow('503');
});

it('isolates both query caches by API origin and workspace', () => {
  const first = makeHost();
  const second = { ...first, workspaceId: 'other-workspace' };
  expect(useProjects(first)).toMatchObject({ queryKey: ['phoenix', first.apiBaseUrl, WORKSPACE, 'projects'] });
  expect(useProjects(second)).not.toEqual(useProjects(first));
  expect(useSpans(first, 'demo')).toMatchObject({
    queryKey: ['phoenix', first.apiBaseUrl, WORKSPACE, 'spans', 'demo'],
  });
  expect(useSpans(second, 'demo')).not.toEqual(useSpans(first, 'demo'));
});
