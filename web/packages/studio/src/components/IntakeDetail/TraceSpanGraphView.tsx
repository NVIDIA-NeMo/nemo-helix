// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FeedbackAnnotationInputValue } from '@nemo/sdk/generated/platform/schema';
import {
  Button,
  Flex,
  SegmentedControl,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { ResizeablePanel } from '@studio/components/common/ResizeablePanel';
import { DagCanvas } from '@studio/components/DagCanvas';
import {
  buildTraceGraph,
  getMostTokensSpan,
  getTraceGraphDisplayName,
  type TraceGraphMode,
} from '@studio/components/IntakeDetail/traceGraph';
import { TraceSelectedSpanPanel } from '@studio/components/IntakeDetail/TraceSelectedSpanPanel';
import { formatDateTime } from '@studio/util/date';
import type { SpanTableRow } from '@studio/util/intakeTelemetry';
import { type FC, type ReactNode, useEffect, useMemo, useState } from 'react';

interface TraceSpanGraphViewProps {
  spanRows: SpanTableRow[];
  selectedSpan: SpanTableRow | undefined;
  traceId: string;
  workspace: string;
  onSelectSpan: (spanId: string, traceId: string) => void;
  banner: ReactNode;
  expandToken: number;
  collapseToken: number;
  activeFeedback?: FeedbackAnnotationInputValue;
  annotationCount?: number;
  hasNotes?: boolean;
  focusNoteNonce?: number;
  onAddNote: () => void;
  emptyContent?: ReactNode;
}

const readGraphMode = (key: string): TraceGraphMode => {
  try {
    return sessionStorage.getItem(key) === 'all' ? 'all' : 'grouped';
  } catch {
    return 'grouped';
  }
};

const storeGraphMode = (key: string, mode: TraceGraphMode): void => {
  try {
    sessionStorage.setItem(key, mode);
  } catch {
    // Storage can be unavailable in privacy-restricted browser contexts.
  }
};

export const TraceSpanGraphView: FC<TraceSpanGraphViewProps> = ({
  spanRows,
  selectedSpan,
  traceId,
  workspace,
  onSelectSpan,
  banner,
  expandToken,
  collapseToken,
  activeFeedback,
  annotationCount,
  hasNotes,
  focusNoteNonce,
  onAddNote,
  emptyContent,
}) => {
  const modeStorageKey = `nemo-studio:trace-graph-mode:${workspace}:${traceId}`;
  const [graphMode, setGraphMode] = useState<TraceGraphMode>(() => readGraphMode(modeStorageKey));
  const [showLongestPath, setShowLongestPath] = useState(false);
  const [tokenFocus, setTokenFocus] = useState<{ spanId: string; nonce: number }>();
  useEffect(() => storeGraphMode(modeStorageKey, graphMode), [graphMode, modeStorageKey]);
  const graph = useMemo(
    () => buildTraceGraph(spanRows, graphMode, { highlightLongestPath: showLongestPath }),
    [graphMode, showLongestPath, spanRows]
  );
  const longestPathNodeIds = useMemo(
    () => graph.nodes.filter(({ data }) => data.highlighted).map(({ id }) => id),
    [graph.nodes]
  );
  const mostTokensSpan = useMemo(() => getMostTokensSpan(spanRows), [spanRows]);
  const selectedNodeId = selectedSpan ? graph.nodeBySpanId.get(selectedSpan.span_id) : undefined;
  const selectedGroup = selectedNodeId ? (graph.spansByNode.get(selectedNodeId) ?? []) : [];
  const selectedCallIndex = selectedSpan
    ? selectedGroup.findIndex(({ span_id }) => span_id === selectedSpan.span_id)
    : -1;
  const selectedSpanForPanel = useMemo(
    () =>
      selectedSpan ? { ...selectedSpan, name: getTraceGraphDisplayName(selectedSpan) } : undefined,
    [selectedSpan]
  );
  const selectedUnit =
    selectedGroup[0]?.kind === 'TOOL' || selectedGroup[0]?.kind === 'LLM' ? 'Call' : 'Span';
  const callLabel = (span: SpanTableRow, index: number): string =>
    `${selectedUnit} ${index + 1} · ${formatDateTime(span.started_at, false)} · ${span.status}`;
  const graphSummary =
    graphMode === 'grouped'
      ? `${graph.nodes.length.toLocaleString()} group${graph.nodes.length === 1 ? '' : 's'} from ${spanRows.length.toLocaleString()} span${spanRows.length === 1 ? '' : 's'}`
      : `${graph.nodes.length.toLocaleString()} span${graph.nodes.length === 1 ? '' : 's'} and ${graph.edges.length.toLocaleString()} parent link${graph.edges.length === 1 ? '' : 's'}`;
  const graphPanel = (
    <Stack className="h-full min-w-0 overflow-hidden bg-surface-sunken">
      <Flex
        align="center"
        justify="between"
        gap="density-lg"
        className="border-b border-base bg-surface-raised px-density-md py-density-sm"
      >
        <Flex align="center" gap="density-sm">
          <SegmentedControl
            size="tiny"
            value={graphMode}
            onValueChange={(value) => setGraphMode(value as TraceGraphMode)}
            items={[
              { value: 'grouped', children: 'Grouped' },
              { value: 'all', children: 'All spans' },
            ]}
          />
          {graphMode === 'all' ? (
            <Flex align="center" gap="density-xs">
              <Button
                kind="tertiary"
                size="tiny"
                color="neutral"
                aria-pressed={showLongestPath}
                title="Highlight the root-to-leaf path containing the most spans"
                className={`!h-5 !min-h-5 rounded-full !border !border-base !px-2 !py-0 !text-[11px] !font-normal ${
                  showLongestPath
                    ? '!border-strong !bg-surface-sunken text-primary'
                    : '!bg-surface-raised text-secondary'
                }`}
                onClick={() => setShowLongestPath((current) => !current)}
              >
                Longest path
              </Button>
              {mostTokensSpan ? (
                <Button
                  kind="tertiary"
                  size="tiny"
                  color="neutral"
                  aria-pressed={selectedSpan?.span_id === mostTokensSpan.span_id}
                  title="Select the individual LLM span with the highest recorded token count"
                  className={`!h-5 !min-h-5 rounded-full !border !border-base !px-2 !py-0 !text-[11px] !font-normal ${
                    selectedSpan?.span_id === mostTokensSpan.span_id
                      ? '!border-strong !bg-surface-sunken text-primary'
                      : '!bg-surface-raised text-secondary'
                  }`}
                  onClick={() => {
                    onSelectSpan(mostTokensSpan.span_id, mostTokensSpan.trace_id ?? traceId);
                    setTokenFocus((current) => ({
                      spanId: mostTokensSpan.span_id,
                      nonce: (current?.nonce ?? 0) + 1,
                    }));
                  }}
                >
                  Most tokens
                </Button>
              ) : null}
            </Flex>
          ) : null}
        </Flex>
        <Text kind="body/regular/xs" className="text-secondary">
          {graphSummary}
        </Text>
      </Flex>
      <div className="h-[min(38rem,calc(100vh-16rem))] min-h-[28rem] w-full">
        <DagCanvas
          nodes={graph.nodes}
          edges={graph.edges}
          direction="LR"
          selectedNodeId={selectedNodeId}
          fitNodeIds={showLongestPath ? longestPathNodeIds : undefined}
          centerNodeId={tokenFocus?.spanId}
          centerNodeNonce={tokenFocus?.nonce}
          viewportStorageKey={`nemo-studio:trace-graph:${workspace}:${traceId}`}
          onNodeClick={(nodeId) => {
            const groupedSpans = graph.spansByNode.get(nodeId) ?? [];
            const span =
              groupedSpans.find(({ span_id }) => span_id === selectedSpan?.span_id) ??
              groupedSpans[0];
            if (span) onSelectSpan(span.span_id, span.trace_id ?? traceId);
          }}
        />
      </div>
    </Stack>
  );

  const detailPanel = (
    <Stack gap="density-sm" className="min-w-0">
      {graphMode === 'grouped' && selectedSpan && selectedGroup.length > 1 ? (
        <Flex
          align="center"
          justify="between"
          gap="density-md"
          className="border-b border-base bg-surface-raised px-density-lg py-density-sm"
        >
          <Text kind="body/semibold/xs">
            {selectedGroup.length.toLocaleString()} {selectedUnit.toLowerCase()}s
          </Text>
          <SelectRoot
            size="small"
            value={selectedSpan.span_id}
            onValueChange={(spanId: string) => onSelectSpan(spanId, traceId)}
          >
            <SelectTrigger
              aria-label={`Selected ${selectedUnit.toLowerCase()}`}
              className="w-56"
              renderValue={() =>
                selectedSpan && selectedCallIndex >= 0
                  ? callLabel(selectedSpan, selectedCallIndex)
                  : undefined
              }
            />
            <SelectContent className="min-w-56">
              <SelectListbox>
                {selectedGroup.map((span, index) => (
                  <SelectItem key={span.span_id} value={span.span_id}>
                    {callLabel(span, index)}
                  </SelectItem>
                ))}
              </SelectListbox>
            </SelectContent>
          </SelectRoot>
        </Flex>
      ) : null}
      <TraceSelectedSpanPanel
        workspace={workspace}
        selectedSpan={selectedSpanForPanel}
        banner={banner}
        expandToken={expandToken}
        collapseToken={collapseToken}
        activeFeedback={activeFeedback}
        annotationCount={annotationCount}
        hasNotes={hasNotes}
        focusNoteNonce={focusNoteNonce}
        onAddNote={onAddNote}
        emptyContent={emptyContent}
      />
    </Stack>
  );

  return (
    <ResizeablePanel
      className="min-h-[28rem] min-w-0"
      defaultLeftWidth={760}
      minLeftWidth={480}
      minRightWidth={352}
      leftClassName="overflow-hidden"
      rightClassName="overflow-y-auto"
      slotLeft={graphPanel}
      slotRight={detailPanel}
    />
  );
};
