// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Connection, OTLPSnippet, PluginHost, UISession } from './types';

const base = (host: PluginHost) =>
  `${host.apiBaseUrl}/apis/observability/v2/workspaces/${encodeURIComponent(host.workspaceId)}`;

async function call<T>(host: PluginHost, path: string, method = 'GET'): Promise<T> {
  // Token per request, so calls survive OIDC silent renew.
  const token = host.auth.getAccessToken();
  const response = await fetch(`${base(host)}${path}`, {
    method,
    // The session response sets the frame's cookie; it must be stored.
    credentials: 'same-origin',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    const detail = await response.json().then((body) => body?.detail, () => undefined);
    throw new Error(typeof detail === 'string' ? detail : `${method} ${path} failed (${response.status})`);
  }
  return (await response.json()) as T;
}

export const listConnections = (host: PluginHost) =>
  call<{ data: Connection[] }>(host, '/connections').then((body) => body.data);

export const createSession = (host: PluginHost, name: string) =>
  call<UISession>(host, `/connections/${encodeURIComponent(name)}/session`, 'POST');

export const getOtlpSnippet = (host: PluginHost, name: string) =>
  call<OTLPSnippet>(host, `/connections/${encodeURIComponent(name)}/otlp-snippet`);
