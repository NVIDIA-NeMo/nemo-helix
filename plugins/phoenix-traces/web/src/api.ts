// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fromPhoenix, type PhoenixSpan, type Span } from './traces';
import type { PluginHost } from './types';

// The adapter address and credentials belong to Helix's manifest/secret registry,
// never to the browser. Workspace must be in the path for gateway authorization.
const BASE = '/apis/plugins/phoenix/v2/workspaces';
const PAGE_LIMIT = 1000;
const MAX_PAGES = 10;
export type ApiHost = Pick<PluginHost, 'apiBaseUrl' | 'workspaceId' | 'auth'>;

interface Page<T> {
  data: T[];
  next_cursor: string | null;
}

async function get<T>(host: ApiHost, path: string, signal?: AbortSignal): Promise<T> {
  const token = host.auth.getAccessToken();
  const res = await fetch(
    `${host.apiBaseUrl.replace(/\/$/, '')}${BASE}/${encodeURIComponent(host.workspaceId)}${path}`,
    {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      redirect: 'error',
      signal,
    }
  );
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function allPages<T>(
  host: ApiHost,
  path: string,
  params: Record<string, string>,
  signal?: AbortSignal
): Promise<T[]> {
  const items: T[] = [];
  let cursor: string | null = null;
  for (let page = 0; page < MAX_PAGES; page++) {
    const query = new URLSearchParams({ ...params, limit: String(PAGE_LIMIT) });
    if (cursor) query.set('cursor', cursor);
    const result: Page<T> = await get(host, `${path}?${query}`, signal);
    items.push(...result.data);
    cursor = result.next_cursor;
    if (!cursor) return items;
  }
  // A partial project can invent roots and incorrect trace/token counts.
  throw new Error('Project exceeds the demo viewer pagination limit; use Intake to narrow the query.');
}

export const listProjects = async (host: ApiHost, signal?: AbortSignal): Promise<string[]> =>
  (await allPages<{ name: string }>(host, '/projects', {}, signal)).map((project) => project.name);

export const listSpans = async (
  host: ApiHost,
  project: string,
  signal?: AbortSignal
): Promise<Span[]> =>
  (await allPages<PhoenixSpan>(host, '/spans', { project }, signal)).map(fromPhoenix);
