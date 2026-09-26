// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Kern plugin API helpers.
 *
 * ONE constant for the route base — the integrator can fix it in one line.
 * All helpers call host.auth.getAccessToken() per request so calls keep
 * working after OIDC silent renew rotates the token.
 */

// Route base path for the kern plugin backend.
// Mirrors ExampleService.name="example" → /apis/example; kern → /apis/kern.
// The integrator adjusts this single constant if the mount prefix ever changes.
export const KERN_API_BASE = "/apis/kern/v1/workspaces";

// --------------------------------------------------------------------------
// Wire types (mirror the backend contract shapes)
// --------------------------------------------------------------------------

export interface KernFlag {
  ulid: string;
  severity: "low" | "medium" | "high";
  audience: string;
  reason: string;
  session_id: string | null;
  created: string;        // ISO-8601
  verdict: string | null;
  acked: boolean;
}

export interface KernFailure {
  ulid: string;
  kind: string;
  category: string;
  created: string;        // ISO-8601
  summary: string;
}

export interface KernDecision {
  ts: string;             // ISO-8601
  rule: string | null;
  decision: string;
  action: string;
  stage: string;
  latency_ms: number | null;
  reason: string | null;
  session_id: string | null;
}

export interface KernMemoryEntry {
  ulid: string;
  kind: string;
  tags: string[];
  scope: string;
  created: string;        // ISO-8601
  title: string;
}

export interface KernMemoryDetail extends KernMemoryEntry {
  body: string;
}

export interface KernSession {
  agent: string;
  model: string | null;
  status: "working" | "idle";
  ctx_fill: number | null;  // 0..1
  cost: number;
  hold: string | null;
  last_seen: string;        // ISO-8601
}

// --------------------------------------------------------------------------
// Internal helpers
// --------------------------------------------------------------------------

/** Compose a URL with optional query params, skipping undefined values. */
function buildUrl(
  apiBaseUrl: string,
  workspaceId: string,
  path: string,
  params?: Record<string, string | number | undefined>,
): string {
  const base = `${apiBaseUrl}${KERN_API_BASE}/${workspaceId}${path}`;
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== "") q.set(k, String(v));
  }
  const qs = q.toString();
  return qs ? `${base}?${qs}` : base;
}

/** Thin fetch wrapper — attaches Bearer token when present, throws on non-2xx. */
async function apiFetch<T>(
  url: string,
  getToken: () => string,
  init?: RequestInit,
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(url, { ...init, headers: { ...headers, ...(init?.headers as Record<string, string> | undefined) } });
  if (!res.ok) {
    throw new Error(`Kern API error: HTTP ${res.status} ${res.statusText} — ${url}`);
  }
  return res.json() as Promise<T>;
}

// --------------------------------------------------------------------------
// Public API helpers
// --------------------------------------------------------------------------

/** List attention flags. */
export function listFlags(
  apiBaseUrl: string,
  workspaceId: string,
  getToken: () => string,
  opts?: { limit?: number; severity?: string },
): Promise<KernFlag[]> {
  return apiFetch<KernFlag[]>(
    buildUrl(apiBaseUrl, workspaceId, "/flags", {
      limit: opts?.limit,
      severity: opts?.severity,
    }),
    getToken,
  );
}

/** Ack a flag — POST …/flags/{ulid}/ack. */
export function ackFlag(
  apiBaseUrl: string,
  workspaceId: string,
  getToken: () => string,
  ulid: string,
): Promise<Record<string, never>> {
  return apiFetch<Record<string, never>>(
    buildUrl(apiBaseUrl, workspaceId, `/flags/${ulid}/ack`),
    getToken,
    { method: "POST" },
  );
}

/** List failure memory entries. */
export function listFailures(
  apiBaseUrl: string,
  workspaceId: string,
  getToken: () => string,
  opts?: { limit?: number },
): Promise<KernFailure[]> {
  return apiFetch<KernFailure[]>(
    buildUrl(apiBaseUrl, workspaceId, "/failures", { limit: opts?.limit }),
    getToken,
  );
}

/** List recent gate decisions. */
export function listDecisions(
  apiBaseUrl: string,
  workspaceId: string,
  getToken: () => string,
  opts?: { limit?: number },
): Promise<KernDecision[]> {
  return apiFetch<KernDecision[]>(
    buildUrl(apiBaseUrl, workspaceId, "/decisions", { limit: opts?.limit }),
    getToken,
  );
}

/** List kmem memory entries (with optional free-text search). */
export function listMemory(
  apiBaseUrl: string,
  workspaceId: string,
  getToken: () => string,
  opts?: { limit?: number; q?: string },
): Promise<KernMemoryEntry[]> {
  return apiFetch<KernMemoryEntry[]>(
    buildUrl(apiBaseUrl, workspaceId, "/memory", {
      limit: opts?.limit,
      q: opts?.q,
    }),
    getToken,
  );
}

/** Fetch a single kmem memory entry (with full body). */
export function getMemory(
  apiBaseUrl: string,
  workspaceId: string,
  getToken: () => string,
  ulid: string,
): Promise<KernMemoryDetail> {
  return apiFetch<KernMemoryDetail>(
    buildUrl(apiBaseUrl, workspaceId, `/memory/${ulid}`),
    getToken,
  );
}

/** List live sessions. */
export function listSessions(
  apiBaseUrl: string,
  workspaceId: string,
  getToken: () => string,
): Promise<KernSession[]> {
  return apiFetch<KernSession[]>(
    buildUrl(apiBaseUrl, workspaceId, "/sessions"),
    getToken,
  );
}
