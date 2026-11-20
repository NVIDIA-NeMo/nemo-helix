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
  /** True when the server's CHRISTMAS_OVERRIDE decided the answer instead of the calendar. */
  forced: boolean;
}

/** The viewer's own calendar date (not UTC), as YYYY-MM-DD. */
export function localDate(now: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

/** A plugin route through the Helix gateway: Studio -> /apis/plugins/<name>/... -> the Rust server. */
const pluginUrl = (host: ApiHost, route: string): string =>
  `${host.apiBaseUrl.replace(/\/$/, '')}/apis/plugins/${PLUGIN_NAME}/v2/workspaces/` +
  `${encodeURIComponent(host.workspaceId)}/${route}`;

export async function getAnswer(host: ApiHost, date: string, signal?: AbortSignal): Promise<Answer> {
  const token = host.auth.getAccessToken();
  const url = pluginUrl(host, `is-it-christmas?date=${encodeURIComponent(date)}`);
  const res = await fetch(url, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    redirect: 'error',
    signal,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as Answer;
}

/** What the Rust server reports when its handler panics. */
export interface RustPanic {
  /** The text Rust prints to stderr, including the RUST_BACKTRACE note. */
  detail: string;
  kind: 'panic';
  thread: string;
  location: string | null;
  message: string;
}

/**
 * Calls the endpoint that panics on purpose. A real panic comes back as HTTP 500 with its
 * report as JSON, so a 500 is the expected answer here and anything else is the surprise.
 */
export async function getPanic(host: ApiHost, signal?: AbortSignal): Promise<RustPanic> {
  const token = host.auth.getAccessToken();
  const res = await fetch(pluginUrl(host, 'panic'), {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    redirect: 'error',
    signal,
  });
  if (res.status === 500) return (await res.json()) as RustPanic;
  throw new Error(res.ok ? 'Expected the endpoint to panic, but it answered normally' : `${res.status} ${res.statusText}`);
}

/** An image the Rust server wraps in JSON as a data URL, so the gateway relays it as-is. */
export interface Picture {
  name: string;
  content_type: string;
  bytes: number;
  data_url: string;
}

export type PictureKind = 'stefan' | 'stefan-animated';
export type Mood = 'happy' | 'sad';

export async function getPicture(
  host: ApiHost,
  kind: PictureKind,
  { mood, signal }: { mood?: Mood; signal?: AbortSignal } = {}
): Promise<Picture> {
  const token = host.auth.getAccessToken();
  const res = await fetch(pluginUrl(host, mood ? `${kind}?mood=${mood}` : kind), {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    redirect: 'error',
    signal,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as Picture;
}
