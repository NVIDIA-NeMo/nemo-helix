// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SpanStatus } from '@nemo/sdk/generated/platform/schema';
import {
  TreeNavBranch,
  TreeNavBranchTrigger,
  TreeNavLeaf,
  TreeNavList,
  TreeNavRoot,
  Tooltip,
} from '@nvidia/foundations-react-core';
import { getSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/registry';
import {
  getSpanKindColorClass,
  getSpanKindConfig,
} from '@studio/components/SpanKindBadge/spanKindConfig';
import {
  formatDurationMs,
  getTraceDisplayName,
  getSpanDisplayName,
  getSpanDurationMs,
  type SessionTrajectory,
  type SpanTreeNode,
} from '@studio/util/intakeTelemetry';
import { MessagesSquare, TriangleAlert, Workflow } from 'lucide-react';
import { type FC, type ReactNode, useState } from 'react';

interface TraceSpanTreeProps {
  /** Session traces and their summary span trajectories. */
  trajectories: SessionTrajectory[];
  /** Trace selected in the URL. */
  activeTraceId?: string;
  /** Overall trace duration shown on the leading "Session" row. */
  sessionDurationMs?: number;
  /** Marks the "Session" row as errored when the trace failed. */
  sessionErrored?: boolean;
  /** Span currently highlighted in the tree (mirrors the open accordion). */
  activeSpanId: string | null;
  /** Fired when a span row is activated; opens/scrolls the matching accordion. */
  onSelectSpan: (spanId: string, traceId: string) => void;
  /** Fired when a trace row is activated. */
  onSelectTrace?: (traceId: string) => void;
  /** Fired when the leading "Session" row is activated; returns to the session summary. */
  onSelectSession?: () => void;
  /** Highlights the leading Session row while its detail panel is visible. */
  sessionActive?: boolean;
}

// Tree rows: round the row highlight and add the vertical padding KUI omits.
const ROW_CLASS = 'rounded-[var(--radius-md)] py-[2px]';

const hierarchyTitle = (node: SpanTreeNode): string | undefined => {
  if (node.hierarchyStatus === 'parent_outside_page') return 'Parent span is outside this page';
  if (node.hierarchyStatus === 'cycle_or_unreachable') return 'Unresolved span hierarchy';
  return undefined;
};

interface SpanTreeLabelProps {
  name: string;
  durationMs?: number;
  errored?: boolean;
}

const SpanTreeName: FC<{ name: string }> = ({ name }) => {
  const [tooltipOpen, setTooltipOpen] = useState(false);
  return (
    <Tooltip
      side="right"
      align="start"
      className="max-w-[32rem] break-words"
      slotContent={tooltipOpen ? name : null}
      onOpenChange={setTooltipOpen}
    >
      <span className="flex-1 min-w-0 truncate">{name}</span>
    </Tooltip>
  );
};

/** Shared label layout: name (truncates) + optional error badge + duration. */
const SpanTreeLabel: FC<SpanTreeLabelProps> = ({ name, durationMs, errored }) => (
  <span className="flex flex-1 items-center gap-2 min-w-0">
    <SpanTreeName name={name} />
    {errored && (
      <TriangleAlert
        role="img"
        aria-label="Error"
        fill="currentColor"
        // size-3.5 (14px) + accent-red as utilities, which win over KUI's base-layer svg rule.
        className="size-3.5 text-(--text-color-accent-red)"
      />
    )}
    <span className="shrink-0 font-mono text-(length:--text-12) tabular-nums text-(--text-color-secondary)">
      {formatDurationMs(durationMs)}
    </span>
  </span>
);

const renderSpanNodes = (
  nodes: SpanTreeNode[],
  activeSpanId: string | null,
  onSelectSpan: (spanId: string, traceId: string) => void,
  traceId: string
): ReactNode =>
  nodes.map((node) => {
    const { span, children } = node;
    const kindConfig = getSpanKindConfig(span.kind);
    const Icon = kindConfig.icon;
    const icon = <Icon role="img" aria-hidden className={getSpanKindColorClass(span.kind)} />;
    const active = activeSpanId === span.span_id;
    const title = hierarchyTitle(node);
    // Mirror the accordion/row header: a template may elevate a kind-specific
    // identity (e.g. the evaluator name) over the generic span name.
    const name = getSpanTemplate(span.kind).headerTitle?.(span) ?? getSpanDisplayName(span);
    const label = (
      <SpanTreeLabel
        name={name}
        durationMs={getSpanDurationMs(span)}
        errored={span.status === SpanStatus.error}
      />
    );

    if (children.length > 0) {
      return (
        <TreeNavBranch key={span.span_id} defaultOpen>
          <TreeNavBranchTrigger
            className={ROW_CLASS}
            slotIcon={icon}
            active={active}
            title={title}
            onClick={() => onSelectSpan(span.span_id, traceId)}
          >
            {label}
          </TreeNavBranchTrigger>
          <TreeNavList>
            {renderSpanNodes(children, activeSpanId, onSelectSpan, traceId)}
          </TreeNavList>
        </TreeNavBranch>
      );
    }

    return (
      <TreeNavLeaf
        key={span.span_id}
        className={ROW_CLASS}
        slotIcon={icon}
        active={active}
        title={title}
        onSelect={() => onSelectSpan(span.span_id, traceId)}
      >
        {label}
      </TreeNavLeaf>
    );
  });

const renderTraceNodes = ({
  trajectories,
  activeTraceId,
  activeSpanId,
  onSelectTrace,
  onSelectSpan,
  sessionActive,
}: {
  trajectories: SessionTrajectory[];
  activeTraceId?: string;
  activeSpanId: string | null;
  onSelectTrace?: (traceId: string) => void;
  onSelectSpan: (spanId: string, traceId: string) => void;
  sessionActive?: boolean;
}): ReactNode =>
  trajectories.map(({ trace, spanTree }) => {
    const nodes = spanTree;
    if (nodes.length > 0) {
      return (
        <TreeNavBranch key={trace.id} defaultOpen>
          <TreeNavBranchTrigger
            className={ROW_CLASS}
            slotIcon={<Workflow role="img" aria-hidden />}
            active={trace.id === activeTraceId && !sessionActive && activeSpanId === null}
            title="View trace"
            onClick={() => onSelectTrace?.(trace.id)}
          >
            <SpanTreeLabel
              name={`Trace: ${getTraceDisplayName(trace)}`}
              durationMs={trace.duration_ms}
              errored={trace.status === SpanStatus.error}
            />
          </TreeNavBranchTrigger>
          <TreeNavList>
            {renderSpanNodes(
              nodes,
              trace.id === activeTraceId ? activeSpanId : null,
              onSelectSpan,
              trace.id
            )}
          </TreeNavList>
        </TreeNavBranch>
      );
    }

    return (
      <TreeNavLeaf
        key={trace.id}
        className={ROW_CLASS}
        slotIcon={<Workflow role="img" aria-hidden />}
        active={trace.id === activeTraceId}
        title="View trace"
        onSelect={() => onSelectTrace?.(trace.id)}
      >
        <SpanTreeLabel
          name={`Trace: ${getTraceDisplayName(trace)}`}
          durationMs={trace.duration_ms}
          errored={trace.status === SpanStatus.error}
        />
      </TreeNavLeaf>
    );
  });

/**
 * Hierarchical view of a trace's trajectory, rendered beside the span
 * accordions. Selecting a span highlights it here and opens/scrolls its
 * accordion; opening an accordion highlights the matching tree row.
 */
export const TraceSpanTree: FC<TraceSpanTreeProps> = ({
  trajectories,
  activeTraceId,
  sessionDurationMs,
  sessionErrored,
  activeSpanId,
  onSelectSpan,
  onSelectTrace,
  onSelectSession,
  sessionActive,
}) => (
  <TreeNavRoot aria-label="Trace trajectory" className="w-full">
    <TreeNavList>
      <TreeNavLeaf
        className={ROW_CLASS}
        slotIcon={<MessagesSquare role="img" aria-hidden />}
        active={sessionActive}
        title="View session"
        onSelect={onSelectSession}
      >
        <SpanTreeLabel name="Session" durationMs={sessionDurationMs} errored={sessionErrored} />
      </TreeNavLeaf>
      {renderTraceNodes({
        trajectories,
        activeTraceId,
        activeSpanId,
        onSelectTrace,
        onSelectSpan,
        sessionActive,
      })}
    </TreeNavList>
  </TreeNavRoot>
);
