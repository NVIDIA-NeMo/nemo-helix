// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';
import { Navigate, NavLink, Route, Routes, useParams } from 'react-router';
import { createSession, getOtlpSnippet, listConnections } from './api';
import { pluginPath } from './paths';
import type { Connection, PluginHost, PluginRootProps } from './types';

// Re-mint this long before the cookie expires so the frame never sees a 401.
const RENEW_MARGIN_MS = 60_000;
const MIN_RENEW_DELAY_MS = 30_000;

export function Root({ host }: PluginRootProps) {
  const connections = useQuery({
    queryKey: ['observability', host.workspaceId, 'connections'],
    queryFn: () => listConnections(host),
  });

  if (connections.isPending) return <Message text="Loading trace explorers…" />;
  if (connections.isError) return <Message text={`Could not load trace explorers: ${connections.error.message}`} />;
  const list = connections.data;
  if (list.length === 0) {
    return (
      <Message text="No trace explorer is configured. An operator adds one under observability.connections in the platform config; see the plugin README." />
    );
  }
  return (
    <Routes>
      <Route index element={<Navigate to={pluginPath(host.workspaceId, list[0].name)} replace />} />
      <Route path=":connection" element={<ExplorerPage host={host} connections={list} />} />
    </Routes>
  );
}

function ExplorerPage({ host, connections }: { host: PluginHost; connections: Connection[] }) {
  const { connection: name } = useParams();
  const connection = connections.find((c) => c.name === name);
  const { ready, error } = useUISession(host, connection?.name);
  const [showExport, setShowExport] = useState(false);
  const { set: setBreadcrumbs } = host.breadcrumbs;

  useEffect(() => {
    setBreadcrumbs([
      { label: 'Trace Explorer', href: pluginPath(host.workspaceId) },
      { label: connection?.display_name ?? name ?? '' },
    ]);
    return () => setBreadcrumbs([]);
  }, [setBreadcrumbs, host.workspaceId, connection?.display_name, name]);

  if (!connection) return <Message text={`Unknown trace explorer "${name}".`} />;
  const frameUrl = `${host.apiBaseUrl}${connection.frame_url}`;

  return (
    <Stack gap="2" className="h-full p-4">
      <Flex gap="3" align="center" className="border-b border-subtle pb-2">
        {connections.length > 1 ? (
          connections.map((c) => (
            <NavLink
              key={c.name}
              to={pluginPath(host.workspaceId, c.name)}
              className={({ isActive }) =>
                `px-3 py-1 rounded text-sm font-medium ${isActive ? 'text-primary bg-surface-hover' : 'text-subtle hover:text-primary'}`
              }
            >
              {c.display_name}
            </NavLink>
          ))
        ) : (
          <Text kind="title/sm">{connection.display_name}</Text>
        )}
        <Flex gap="2" className="ml-auto">
          <Button kind="tertiary" onClick={() => setShowExport((v) => !v)}>
            {showExport ? 'Hide agent export' : 'Send agent traces here'}
          </Button>
          {/* Same origin, so the session cookie also authorizes the standalone tab. */}
          <Button kind="secondary" disabled={!ready} onClick={() => window.open(frameUrl, '_blank', 'noopener')}>
            Open in new tab
          </Button>
        </Flex>
      </Flex>
      {showExport && <ExportPanel host={host} connection={connection} />}
      {error ? (
        <Message text={`Could not open ${connection.display_name}: ${error}`} />
      ) : ready ? (
        <iframe
          key={connection.name}
          src={frameUrl}
          title={connection.display_name}
          className="flex-1 w-full rounded border border-subtle bg-surface-base"
        />
      ) : (
        <Message text={`Opening ${connection.display_name}…`} />
      )}
    </Stack>
  );
}

/**
 * Mints the frame's session cookie, then re-mints before it expires. A single in-flight
 * request is shared, so overlapping renewals never race each other.
 */
function useUISession(host: PluginHost, name: string | undefined) {
  const hostRef = useRef(host);
  hostRef.current = host;
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!name) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let inflight: Promise<void> | null = null;
    setReady(false);
    setError(null);

    const mint = (): Promise<void> => {
      inflight ??= createSession(hostRef.current, name)
        .then((session) => {
          if (cancelled) return;
          setReady(true);
          setError(null);
          const delay = Math.max(MIN_RENEW_DELAY_MS, session.expires_at * 1000 - Date.now() - RENEW_MARGIN_MS);
          timer = setTimeout(mint, delay);
        })
        .catch((e: unknown) => {
          if (!cancelled) setError(e instanceof Error ? e.message : String(e));
        })
        .finally(() => {
          inflight = null;
        });
      return inflight;
    };

    void mint();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [name]);

  return { ready, error };
}

function ExportPanel({ host, connection }: { host: PluginHost; connection: Connection }) {
  const snippet = useQuery({
    queryKey: ['observability', host.workspaceId, 'otlp-snippet', connection.name],
    queryFn: () => getOtlpSnippet(host, connection.name),
  });
  if (snippet.isPending) return <Text kind="body/regular/sm">Loading export settings…</Text>;
  if (snippet.isError) return <Text kind="body/regular/sm">{snippet.error.message}</Text>;
  const copy = () =>
    navigator.clipboard.writeText(snippet.data.yaml).then(
      () => host.notifications.notify('Copied telemetry stanza', 'success'),
      () => host.notifications.notify('Copy failed; select the text instead', 'error')
    );
  return (
    <Stack gap="2" className="rounded border border-subtle bg-surface-raised p-3">
      <Text kind="body/regular/sm">
        Add this to an agent&apos;s agent.yaml to export its traces straight to {connection.display_name}. Helix keeps
        sending its own telemetry to Intake as well.
      </Text>
      <pre className="overflow-auto rounded bg-surface-sunken p-2 text-sm">{snippet.data.yaml}</pre>
      <Flex gap="2">
        <Button kind="secondary" onClick={copy}>
          Copy
        </Button>
      </Flex>
      <Text kind="body/regular/sm" className="text-subtle">
        {connection.display_name} must be served under {connection.ui_prefix}: {snippet.data.upstream_setup.join(', ')}
      </Text>
    </Stack>
  );
}

function Message({ text }: { text: string }) {
  return (
    <Flex className="h-full p-6" align="center" justify="center">
      <Text kind="body/regular/md" className="text-subtle">
        {text}
      </Text>
    </Flex>
  );
}
