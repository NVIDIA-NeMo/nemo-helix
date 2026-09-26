// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Kern plugin root.
 *
 * Studio renders this component inside its own React tree (under Studio's
 * Router, QueryClient, and KaizenThemeProvider), so the plugin shares those
 * contexts. It uses Studio's router directly (no BrowserRouter) and Studio's
 * design system — KUI components from @nvidia/foundations-react-core plus
 * Studio's semantic token classes — which are theme-aware.
 *
 * Tabs use React state (not sub-routes) — there is only one nav entry and
 * the four panes are purely in-page layout.
 */

import {
  RelativeTime,
  StatusBadge,
  TableEmptyState,
} from "@nemo/common";
import { Badge, Button, Flex, Stack, Text, TextInput } from "@nvidia/foundations-react-core";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router";
import {
  ackFlag,
  getMemory,
  listDecisions,
  listFailures,
  listFlags,
  listMemory,
  listSessions,
  type KernDecision,
  type KernFailure,
  type KernFlag,
  type KernMemoryEntry,
  type KernSession,
} from "./api";
import { intakeTracesPath, pluginPath } from "./paths";
import type { PluginHost, PluginRootProps } from "./types";

// --------------------------------------------------------------------------
// Root router
// --------------------------------------------------------------------------

export function Root({ host }: PluginRootProps) {
  return (
    <Routes>
      <Route
        index
        element={
          <Navigate to={pluginPath(host.workspaceId, "dashboard")} replace />
        }
      />
      <Route path="dashboard" element={<DashboardPage host={host} />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

// --------------------------------------------------------------------------
// Dashboard — four tabs: Attention, Sessions, Decisions, Memory
// --------------------------------------------------------------------------

type Tab = "attention" | "sessions" | "decisions" | "memory";

const TABS: { id: Tab; label: string }[] = [
  { id: "attention", label: "Attention" },
  { id: "sessions", label: "Sessions" },
  { id: "decisions", label: "Decisions" },
  { id: "memory", label: "Memory" },
];

function DashboardPage({ host }: { host: PluginHost }) {
  const [activeTab, setActiveTab] = useState<Tab>("attention");

  const { set: setBreadcrumbs } = host.breadcrumbs;
  const { workspaceId } = host;
  useEffect(() => {
    setBreadcrumbs([
      { label: "Kern", href: pluginPath(workspaceId, "dashboard") },
      { label: "Dashboard" },
    ]);
    return () => setBreadcrumbs([]);
  }, [setBreadcrumbs, workspaceId]);

  const tabClass = (id: Tab) =>
    `px-3 py-1 rounded text-sm font-medium transition-colors ${
      activeTab === id
        ? "text-primary bg-surface-hover"
        : "text-subtle hover:text-primary"
    }`;

  return (
    <Stack gap="4" className="h-full p-4">
      <Flex gap="0" className="items-center justify-between border-b border-subtle pb-2">
        <Text kind="label/bold/lg">Kern Dashboard</Text>
        <Flex gap="2">
          {TABS.map((t) => (
            <button key={t.id} className={tabClass(t.id)} onClick={() => setActiveTab(t.id)}>
              {t.label}
            </button>
          ))}
        </Flex>
      </Flex>

      <div className="flex-1 overflow-auto">
        {activeTab === "attention" && <AttentionPane host={host} />}
        {activeTab === "sessions" && <SessionsPane host={host} />}
        {activeTab === "decisions" && <DecisionsPane host={host} />}
        {activeTab === "memory" && <MemoryPane host={host} />}
      </div>
    </Stack>
  );
}

// --------------------------------------------------------------------------
// Attention pane — flags + failures
// --------------------------------------------------------------------------

/** Map severity → StatusBadge status string. */
function severityStatus(severity: string): string {
  if (severity === "high") return "error";
  if (severity === "medium") return "paused";
  return "ready"; // low
}

function AttentionPane({ host }: { host: PluginHost }) {
  const qc = useQueryClient();
  const { workspaceId, apiBaseUrl, auth } = host;
  const getToken = auth.getAccessToken;

  const flagsQuery = useQuery<KernFlag[], Error>({
    queryKey: ["kern", workspaceId, "flags"],
    queryFn: () => listFlags(apiBaseUrl, workspaceId, getToken, { limit: 50 }),
    staleTime: 30_000,
  });

  const failuresQuery = useQuery<KernFailure[], Error>({
    queryKey: ["kern", workspaceId, "failures"],
    queryFn: () => listFailures(apiBaseUrl, workspaceId, getToken, { limit: 30 }),
    staleTime: 60_000,
  });

  const ackMutation = useMutation({
    mutationFn: (ulid: string) => ackFlag(apiBaseUrl, workspaceId, getToken, ulid),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["kern", workspaceId, "flags"] });
    },
    onError: (err) => {
      host.notifications.notify(`Ack failed: ${String(err)}`, "error");
    },
  });

  return (
    <Stack gap="4">
      {/* Flags section */}
      <Stack gap="2">
        <Text kind="label/bold/md">Attention Flags</Text>
        <Text kind="body/regular/xs" color="secondary">
          Sleep flags raised by the kern monitor — review and ack to clear.
        </Text>

        {flagsQuery.isError ? (
          <TableEmptyState header="Couldn't load flags" />
        ) : flagsQuery.isPending ? (
          <Text kind="body/regular/sm" color="secondary">Loading…</Text>
        ) : !flagsQuery.data?.length ? (
          <TableEmptyState header="No flags" />
        ) : (
          <Stack gap="2">
            {flagsQuery.data.map((flag) => (
              <FlagRow
                key={flag.ulid}
                flag={flag}
                onAck={() => ackMutation.mutate(flag.ulid)}
                acking={ackMutation.isPending && ackMutation.variables === flag.ulid}
              />
            ))}
          </Stack>
        )}
      </Stack>

      {/* Failures section */}
      <Stack gap="2">
        <Text kind="label/bold/md">Failure Memory</Text>
        <Text kind="body/regular/xs" color="secondary">
          Recent kmem entries of kind: failure.
        </Text>

        {failuresQuery.isError ? (
          <TableEmptyState header="Couldn't load failures" />
        ) : failuresQuery.isPending ? (
          <Text kind="body/regular/sm" color="secondary">Loading…</Text>
        ) : !failuresQuery.data?.length ? (
          <TableEmptyState header="No failures recorded" />
        ) : (
          <div className="rounded border border-subtle overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-subtle bg-surface-sunken">
                  <th className="px-3 py-2 text-left">
                    <Text kind="label/bold/xs" color="secondary">Category</Text>
                  </th>
                  <th className="px-3 py-2 text-left">
                    <Text kind="label/bold/xs" color="secondary">Created</Text>
                  </th>
                  <th className="px-3 py-2 text-left">
                    <Text kind="label/bold/xs" color="secondary">Summary</Text>
                  </th>
                </tr>
              </thead>
              <tbody>
                {failuresQuery.data.map((f) => (
                  <tr key={f.ulid} className="border-b border-subtle last:border-0 hover:bg-surface-hover">
                    <td className="px-3 py-2">
                      <Text kind="body/regular/sm">{f.category || f.kind}</Text>
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap">
                      <RelativeTime datetime={f.created} />
                    </td>
                    <td className="px-3 py-2">
                      <Text kind="body/regular/sm" color="secondary">{f.summary}</Text>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Stack>
    </Stack>
  );
}

function FlagRow({
  flag,
  onAck,
  acking,
}: {
  flag: KernFlag;
  onAck: () => void;
  acking: boolean;
}) {
  return (
    <div className="rounded border border-subtle bg-surface-raised p-3">
      <Flex gap="3" className="items-start justify-between">
        <Stack gap="1" className="flex-1 min-w-0">
          <Flex gap="2" className="items-center flex-wrap">
            <StatusBadge status={severityStatus(flag.severity)} label={flag.severity} />
            {flag.acked && (
              <Badge color="green" kind="outline">acked</Badge>
            )}
            <Text kind="label/bold/xs" color="secondary">{flag.audience}</Text>
            {flag.created && (
              <Text kind="body/regular/xs" color="secondary">
                <RelativeTime datetime={flag.created} />
              </Text>
            )}
          </Flex>
          <Text kind="body/regular/sm">{flag.reason}</Text>
          {flag.verdict && (
            <Text kind="body/regular/xs" color="secondary">Verdict: {flag.verdict}</Text>
          )}
          {flag.session_id && (
            <Text kind="body/regular/xs" color="secondary">Session: {flag.session_id}</Text>
          )}
        </Stack>
        {!flag.acked && (
          <Button kind="secondary" size="sm" onClick={onAck} isLoading={acking}>
            Ack
          </Button>
        )}
      </Flex>
    </div>
  );
}

// --------------------------------------------------------------------------
// Sessions pane — live agent sessions, auto-refresh 5 s
// --------------------------------------------------------------------------

function SessionsPane({ host }: { host: PluginHost }) {
  const { workspaceId, apiBaseUrl, auth } = host;

  const sessionsQuery = useQuery<KernSession[], Error>({
    queryKey: ["kern", workspaceId, "sessions"],
    queryFn: () => listSessions(apiBaseUrl, workspaceId, auth.getAccessToken),
    refetchInterval: 5_000,
    staleTime: 0,
  });

  if (sessionsQuery.isError) {
    return <TableEmptyState header="Couldn't load sessions" />;
  }

  return (
    <Stack gap="2">
      <Flex gap="2" className="items-center">
        <Text kind="label/bold/md">Live Sessions</Text>
        <Text kind="body/regular/xs" color="secondary">(auto-refreshes every 5 s)</Text>
      </Flex>

      {sessionsQuery.isPending ? (
        <Text kind="body/regular/sm" color="secondary">Loading…</Text>
      ) : !sessionsQuery.data?.length ? (
        <TableEmptyState header="No active sessions" />
      ) : (
        <div className="rounded border border-subtle overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-subtle bg-surface-sunken">
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Agent</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Model</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Status</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Ctx fill</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Cost</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Hold</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Updated</Text>
                </th>
              </tr>
            </thead>
            <tbody>
              {sessionsQuery.data.map((s) => (
                <SessionRow key={s.agent} session={s} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Stack>
  );
}

function SessionRow({ session: s }: { session: KernSession }) {
  const ctxPct =
    s.ctx_fill !== null ? Math.min(100, Math.round(s.ctx_fill * 100)) : null;

  return (
    <tr className="border-b border-subtle last:border-0 hover:bg-surface-hover">
      <td className="px-3 py-2 font-mono">
        <Text kind="body/regular/sm">{s.agent}</Text>
      </td>
      <td className="px-3 py-2">
        <Text kind="body/regular/sm" color="secondary">{s.model ?? "—"}</Text>
      </td>
      <td className="px-3 py-2">
        <StatusBadge status={s.status === "working" ? "running" : "ready"} label={s.status} />
      </td>
      <td className="px-3 py-2">
        {ctxPct !== null ? (
          <Flex gap="2" className="items-center">
            <div
              className="h-2 rounded overflow-hidden bg-surface-sunken"
              style={{ width: 64 }}
            >
              <div
                className="h-2 rounded"
                style={{
                  width: `${ctxPct}%`,
                  backgroundColor: ctxPct > 80 ? "var(--color-danger)" : "var(--color-primary)",
                }}
              />
            </div>
            <Text kind="body/regular/xs" color="secondary">{ctxPct}%</Text>
          </Flex>
        ) : (
          <Text kind="body/regular/sm" color="secondary">—</Text>
        )}
      </td>
      <td className="px-3 py-2">
        <Text kind="body/regular/sm">${s.cost.toFixed(4)}</Text>
      </td>
      <td className="px-3 py-2">
        {s.hold ? (
          <Badge color="yellow" kind="solid">{s.hold}</Badge>
        ) : (
          <Text kind="body/regular/sm" color="secondary">—</Text>
        )}
      </td>
      <td className="px-3 py-2 whitespace-nowrap">
        <RelativeTime datetime={s.last_seen} />
      </td>
    </tr>
  );
}

// --------------------------------------------------------------------------
// Decisions pane — recent gate decisions + link to Studio Traces
// --------------------------------------------------------------------------

function DecisionsPane({ host }: { host: PluginHost }) {
  const { workspaceId, apiBaseUrl, auth } = host;

  const decisionsQuery = useQuery<KernDecision[], Error>({
    queryKey: ["kern", workspaceId, "decisions"],
    queryFn: () => listDecisions(apiBaseUrl, workspaceId, auth.getAccessToken, { limit: 100 }),
    staleTime: 15_000,
  });

  const tracesHref = intakeTracesPath(workspaceId);

  return (
    <Stack gap="2">
      <Flex gap="0" className="items-center justify-between">
        <Text kind="label/bold/md">Gate Decisions</Text>
        <Button
          kind="secondary"
          size="sm"
          onClick={() => host.navigation.navigate(tracesHref)}
        >
          View Traces ↗
        </Button>
      </Flex>
      <Text kind="body/regular/xs" color="secondary">
        Recent kern gate decisions — allow/block/hold from the policy engine.
      </Text>

      {decisionsQuery.isError ? (
        <TableEmptyState header="Couldn't load decisions" />
      ) : decisionsQuery.isPending ? (
        <Text kind="body/regular/sm" color="secondary">Loading…</Text>
      ) : !decisionsQuery.data?.length ? (
        <TableEmptyState header="No decisions recorded" />
      ) : (
        <div className="rounded border border-subtle overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-subtle bg-surface-sunken">
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Time</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Rule</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Decision</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Latency</Text>
                </th>
                <th className="px-3 py-2 text-left">
                  <Text kind="label/bold/xs" color="secondary">Reason</Text>
                </th>
              </tr>
            </thead>
            <tbody>
              {decisionsQuery.data.map((d, i) => (
                <DecisionRow key={`${d.ts}-${i}`} decision={d} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Stack>
  );
}

function decisionStatus(decision: string): string {
  if (decision === "allow") return "running";
  if (decision === "block") return "error";
  if (decision === "hold") return "paused";
  return "default";
}

function DecisionRow({ decision: d }: { decision: KernDecision }) {
  return (
    <tr className="border-b border-subtle last:border-0 hover:bg-surface-hover">
      <td className="px-3 py-2 whitespace-nowrap">
        <RelativeTime datetime={d.ts} />
      </td>
      <td className="px-3 py-2 font-mono">
        <Text kind="body/regular/xs">{d.rule ?? "—"}</Text>
      </td>
      <td className="px-3 py-2">
        <StatusBadge status={decisionStatus(d.decision)} label={d.decision} />
      </td>
      <td className="px-3 py-2 whitespace-nowrap">
        <Text kind="body/regular/sm" color="secondary">
          {d.latency_ms !== null ? `${d.latency_ms.toFixed(1)} ms` : "—"}
        </Text>
      </td>
      <td className="px-3 py-2">
        <Text kind="body/regular/sm" color="secondary">{d.reason ?? "—"}</Text>
      </td>
    </tr>
  );
}

// --------------------------------------------------------------------------
// Memory pane — kmem entries with search + expand body
// --------------------------------------------------------------------------

function MemoryPane({ host }: { host: PluginHost }) {
  const { workspaceId, apiBaseUrl, auth } = host;
  const [search, setSearch] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);

  // Debounce search input by 400 ms.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(search), 400);
    return () => clearTimeout(t);
  }, [search]);

  const memoryQuery = useQuery<KernMemoryEntry[], Error>({
    queryKey: ["kern", workspaceId, "memory", debouncedQ],
    queryFn: () =>
      listMemory(apiBaseUrl, workspaceId, auth.getAccessToken, {
        limit: 50,
        q: debouncedQ || undefined,
      }),
    staleTime: 30_000,
  });

  const detailQuery = useQuery({
    queryKey: ["kern", workspaceId, "memory", expanded],
    queryFn: () =>
      expanded
        ? getMemory(apiBaseUrl, workspaceId, auth.getAccessToken, expanded)
        : Promise.resolve(null),
    enabled: expanded !== null,
    staleTime: 60_000,
  });

  const toggleExpand = useCallback(
    (ulid: string) => setExpanded((prev) => (prev === ulid ? null : ulid)),
    [],
  );

  return (
    <Stack gap="2">
      <Text kind="label/bold/md">Memory Store</Text>
      <Text kind="body/regular/xs" color="secondary">
        Recent kmem entries. Click a row to expand the body. Use the search box
        to filter by title or content.
      </Text>

      <TextInput
        value={search}
        onChange={(e) => setSearch(e.currentTarget.value)}
        placeholder="Search memory…"
        aria-label="Search memory"
      />

      {memoryQuery.isError ? (
        <TableEmptyState header="Couldn't load memory" />
      ) : memoryQuery.isPending ? (
        <Text kind="body/regular/sm" color="secondary">Loading…</Text>
      ) : !memoryQuery.data?.length ? (
        <TableEmptyState header={debouncedQ ? `No results for "${debouncedQ}"` : "No memory entries"} />
      ) : (
        <Stack gap="1">
          {memoryQuery.data.map((entry) => (
            <MemoryRow
              key={entry.ulid}
              entry={entry}
              isExpanded={expanded === entry.ulid}
              onToggle={() => toggleExpand(entry.ulid)}
              body={expanded === entry.ulid ? (detailQuery.data?.body ?? null) : null}
              loadingBody={expanded === entry.ulid && detailQuery.isPending}
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
}

function MemoryRow({
  entry,
  isExpanded,
  onToggle,
  body,
  loadingBody,
}: {
  entry: KernMemoryEntry;
  isExpanded: boolean;
  onToggle: () => void;
  body: string | null;
  loadingBody: boolean;
}) {
  return (
    <div className="rounded border border-subtle overflow-hidden">
      <button
        className="w-full text-left px-3 py-2 hover:bg-surface-hover transition-colors"
        onClick={onToggle}
      >
        <Flex gap="2" className="items-start justify-between">
          <Stack gap="1" className="flex-1 min-w-0">
            <Flex gap="2" className="items-center flex-wrap">
              <Badge color="gray" kind="outline">{entry.kind}</Badge>
              {entry.tags.slice(0, 3).map((t) => (
                <Badge key={t} color="teal" kind="outline">{t}</Badge>
              ))}
              {entry.tags.length > 3 && (
                <Text kind="body/regular/xs" color="secondary">+{entry.tags.length - 3}</Text>
              )}
              <Text kind="body/regular/xs" color="secondary">{entry.scope}</Text>
              <Text kind="body/regular/xs" color="secondary">
                <RelativeTime datetime={entry.created} />
              </Text>
            </Flex>
            <Text kind="body/regular/sm">{entry.title}</Text>
          </Stack>
          <Text kind="body/regular/xs" color="secondary">{isExpanded ? "▲" : "▼"}</Text>
        </Flex>
      </button>

      {isExpanded && (
        <div className="border-t border-subtle px-3 py-2 bg-surface-sunken">
          {loadingBody ? (
            <Text kind="body/regular/sm" color="secondary">Loading body…</Text>
          ) : body ? (
            <pre className="text-xs text-subtle font-mono whitespace-pre-wrap overflow-x-auto">
              {body}
            </pre>
          ) : (
            <Text kind="body/regular/sm" color="secondary">No body available.</Text>
          )}
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------
// 404
// --------------------------------------------------------------------------

function NotFound() {
  return (
    <Stack gap="2" className="p-4">
      <Text kind="label/bold/md">Page not found</Text>
      <Text kind="body/regular/sm" color="secondary">
        This path does not exist within the Kern plugin.
      </Text>
    </Stack>
  );
}
