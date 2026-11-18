// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge, TableEmptyState } from '@nemo/common';
import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { useEffect, useMemo, useState } from 'react';
import { useParams, useSearchParams } from 'react-router';
import { InputView, OutputView } from './Formatted';
import { pluginPath } from './paths';
import { useSpans } from './queries';
import { formatDuration, toRows, type Span } from './traces';
import { STATUS_CONFIG } from './TracesPage';
import type { PluginHost } from './types';

const TABS = ['Input', 'Output', 'Raw', 'Attributes'] as const;
type Tab = (typeof TABS)[number];

const KIND_COLOR: Record<string, 'blue' | 'green' | 'purple' | 'yellow' | 'gray'> = {
  LLM: 'purple',
  TOOL: 'green',
  AGENT: 'blue',
  CHAIN: 'blue',
  RETRIEVER: 'yellow',
};

/** Pretty-print JSON strings; leave everything else as-is. */
const pretty = (value: unknown): string => {
  if (typeof value !== 'string') return JSON.stringify(value, null, 2) ?? '';
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
};

export function TraceDetailPage({ host }: { host: PluginHost }) {
  const { traceId } = useParams();
  const [params] = useSearchParams();
  const project = params.get('project') ?? undefined;
  const spans = useSpans(host, project);

  const traceSpans = useMemo(() => (spans.data ?? []).filter((s) => s.traceId === traceId), [spans.data, traceId]);
  const rows = useMemo(() => toRows(traceSpans), [traceSpans]);
  const start = Math.min(...traceSpans.map((s) => s.start));
  const total = Math.max(...traceSpans.map((s) => s.end)) - start || 1;

  const [selectedId, setSelectedId] = useState<string>();
  const selected = traceSpans.find((s) => s.id === selectedId) ?? rows[0]?.span;

  const { set: setBreadcrumbs } = host.breadcrumbs;
  const { workspaceId } = host;
  useEffect(() => {
    setBreadcrumbs([
      { label: 'Phoenix' },
      { label: 'Traces', href: `${pluginPath(workspaceId, 'traces')}?project=${encodeURIComponent(project ?? '')}` },
      { label: (traceId ?? '').slice(0, 12) },
    ]);
    return () => setBreadcrumbs([]);
  }, [setBreadcrumbs, workspaceId, project, traceId]);

  if (spans.isError) {
    return <TableEmptyState header="Couldn't load trace" emptyMessage="The Phoenix plugin API request failed." />;
  }
  if (!spans.isPending && rows.length === 0) {
    return <TableEmptyState header="Trace not found" emptyMessage={`No spans for ${traceId} in ${project ?? 'this project'}.`} />;
  }

  return (
    // Layout-critical sizes are inline: Studio's Tailwind only generates classes Studio
    // itself uses, so arbitrary values (w-[460px]) and colors from here may not exist.
    <div className="flex h-full" style={{ minHeight: 480 }}>
      <aside className="overflow-y-auto border-r border-subtle" style={{ width: 460, flexShrink: 0 }}>
        {rows.map(({ span, depth }) => (
          <button
            key={span.id}
            type="button"
            onClick={() => setSelectedId(span.id)}
            className={`block w-full px-3 py-2 text-left transition-colors ${
              span.id === selected?.id ? 'bg-surface-hover' : 'hover:bg-surface-hover'
            }`}
          >
            <Flex gap="2" align="center" className="pr-2" style={{ paddingLeft: depth * 16 }}>
              <Badge color={KIND_COLOR[span.kind] ?? 'gray'} kind="outline">
                {span.kind}
              </Badge>
              <Text kind="body/regular/sm" className="truncate">
                {span.name}
              </Text>
              <Text kind="body/regular/xs" color="secondary" className="ml-auto shrink-0">
                {formatDuration(span.end - span.start)}
              </Text>
            </Flex>
            <div className="mt-1 rounded bg-surface-sunken" style={{ position: 'relative', height: 4 }}>
              <div
                className="text-primary"
                style={{
                  position: 'absolute',
                  height: 4,
                  borderRadius: 2,
                  background: span.status === 'ERROR' ? '#ef4444' : 'currentColor',
                  left: `${((span.start - start) / total) * 100}%`,
                  width: `${Math.max(((span.end - span.start) / total) * 100, 0.5)}%`,
                }}
              />
            </div>
          </button>
        ))}
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto p-4">{selected ? <SpanDetail span={selected} /> : null}</main>
    </div>
  );
}

function SpanDetail({ span }: { span: Span }) {
  const [tab, setTab] = useState<Tab>('Input');
  const attrs = span.attributes;
  const model = attrs['llm.model_name'];
  const prompt = attrs['llm.token_count.prompt'];
  const completion = attrs['llm.token_count.completion'];

  return (
    <Stack gap="3">
      <Flex gap="2" align="center">
        <Text kind="label/bold/lg">{span.name}</Text>
        <StatusBadge status={span.status} statusConfig={STATUS_CONFIG} />
      </Flex>
      <Text kind="body/regular/sm" color="secondary">
        {span.kind} · {formatDuration(span.end - span.start)}
        {typeof model === 'string' ? ` · ${model}` : ''}
        {typeof prompt === 'number' || typeof completion === 'number'
          ? ` · ${prompt ?? 0} in / ${completion ?? 0} out tokens`
          : ''}
      </Text>
      {span.statusMessage ? (
        <Text kind="body/regular/sm" color="danger">
          {span.statusMessage}
        </Text>
      ) : null}
      <Flex gap="2" className="border-b border-subtle pb-2">
        {TABS.map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => setTab(name)}
            className={`rounded px-3 py-1 text-sm font-medium transition-colors ${
              tab === name ? 'bg-surface-hover text-primary' : 'text-subtle hover:text-primary'
            }`}
          >
            {name}
          </button>
        ))}
      </Flex>
      {tab === 'Input' ? (
        <InputView span={span} />
      ) : tab === 'Output' ? (
        <OutputView span={span} />
      ) : (
        <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded bg-surface-sunken p-3 font-mono text-xs text-subtle">
          {tab === 'Raw'
            ? [attrs['input.value'], attrs['output.value']].map((v) => pretty(v ?? '')).join('\n\n---\n\n')
            : pretty(attrs)}
        </pre>
      )}
    </Stack>
  );
}
