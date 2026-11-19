// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLUGIN_NAME } from './paths';
import type { PluginHost } from './types';

export type ApiHost = Pick<PluginHost, 'apiBaseUrl' | 'workspaceId' | 'auth'>;

export interface Answer {
  date: string;
  is_christmas: boolean;
  days_until: number;
  next_christmas: string;
  message: string;
}

/** The viewer's own calendar date (not UTC), as YYYY-MM-DD. */
export function localDate(now: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

/** Calls the Rust backend through the Helix gateway: Studio -> /apis/plugins/<name>/... -> :6010. */
export async function getAnswer(host: ApiHost, date: string, signal?: AbortSignal): Promise<Answer> {
  const token = host.auth.getAccessToken();
  const url =
    `${host.apiBaseUrl.replace(/\/$/, '')}/apis/plugins/${PLUGIN_NAME}/v2/workspaces/` +
    `${encodeURIComponent(host.workspaceId)}/is-it-christmas?date=${encodeURIComponent(date)}`;
  const res = await fetch(url, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    redirect: 'error',
    signal,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as Answer;
}
