// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fromPhoenix, type PhoenixSpan, type Span } from './traces';
import type { PluginHost } from './types';

// The only URL space the plugin knows: /apis/plugins/<name>/<endpoint>. Where the
// upstream really lives is the platform gateway's business.
const BASE = '/apis/plugins/phoenix';
const PAGE_LIMIT = 1000;
const MAX_PAGES = 10;

interface Page<T> {
  data: T[];
  next_cursor: string | null;
}

async function get<T>(host: PluginHost, path: string): Promise<T> {
  const token = host.auth.getAccessToken();
  const res = await fetch(`${host.apiBaseUrl}${BASE}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const listProjects = async (host: PluginHost): Promise<string[]> =>
  (await get<Page<{ name: string }>>(host, '/projects')).data.map((p) => p.name);

export const listSpans = async (host: PluginHost, project: string): Promise<Span[]> => {
  const spans: Span[] = [];
  let cursor: string | null = null;
  for (let page = 0; page < MAX_PAGES; page++) {
    const query = new URLSearchParams({ limit: String(PAGE_LIMIT) });
    if (cursor) query.set('cursor', cursor);
    const res: Page<PhoenixSpan> = await get(
      host,
      `/projects/${encodeURIComponent(project)}/spans?${query}`
    );
    spans.push(...res.data.map(fromPhoenix));
    cursor = res.next_cursor;
    if (!cursor) break;
  }
  return spans;
};
