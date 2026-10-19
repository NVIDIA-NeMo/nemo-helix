// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CardIconBadge, SelectableCard } from '@studio/components/common/SelectableCard';
import type { DagNodeData, DagNodeStatus } from '@studio/components/DagCanvas/types';
import { Handle, type Node, type NodeProps, Position } from '@xyflow/react';
import { Box } from 'lucide-react';
import { type FC } from 'react';

/** Internal node data: the public {@link DagNodeData} plus the activation callback. */
export interface CardNodeData extends DagNodeData {
  onActivate?: () => void;
  /** Whether an incoming edge terminates here; controls the target handle. Defaults to true. */
  hasIncoming?: boolean;
  /** Whether an outgoing edge starts here; controls the source handle. Defaults to true. */
  hasOutgoing?: boolean;
  /** Prevent card clicks from selecting the React Flow wrapper. */
  stopPropagation?: boolean;
  [key: string]: unknown;
}

export type CardNodeType = Node<CardNodeData, 'card'>;

/** Status → leading icon color, using NVIDIA Foundations feedback tokens. */
const STATUS_ICON_CLASS: Record<DagNodeStatus, string> = {
  idle: 'text-[color:var(--text-color-accent-blue)]',
  running: 'text-[color:var(--text-color-feedback-info)]',
  success: 'text-[color:var(--text-color-feedback-success)]',
  warning: 'text-[color:var(--text-color-feedback-warning)]',
  error: 'text-[color:var(--text-color-feedback-danger)]',
  cancelled: 'text-[color:var(--text-color-feedback-warning)]',
  unknown: 'text-secondary',
};

/**
 * A DAG node rendered as a {@link SelectableCard} — the same card the Data Designer
 * column palette uses — with an icon badge, an accent type label, a description, and
 * variable tags, plus React Flow connection handles on the leading/trailing edges.
 */
export const CardNode: FC<NodeProps<CardNodeType>> = ({
  data,
  selected,
  sourcePosition = Position.Bottom,
  targetPosition = Position.Top,
}) => {
  const {
    title,
    type,
    description,
    tags,
    icon: Icon = Box,
    status = 'idle',
    colorClassName,
    highlighted = false,
    onActivate,
    hasIncoming = true,
    hasOutgoing = true,
    stopPropagation = false,
  } = data;
  const iconClassName = colorClassName ?? STATUS_ICON_CLASS[status];
  const subtitle = [type, status === 'idle' ? null : status].filter(Boolean).join(' · ');
  const emphasisClassName = selected
    ? 'border-brand ring-1 ring-(--color-brand,#76b900)'
    : highlighted
      ? '!border-2 !border-dashed !border-brand'
      : undefined;
  return (
    <>
      {hasIncoming && (
        <Handle type="target" position={targetPosition} className="bg-strong border-none" />
      )}
      <SelectableCard
        title={title}
        ariaLabel={[
          title,
          type,
          status === 'idle' ? null : status,
          highlighted ? 'longest path' : null,
        ]
          .filter(Boolean)
          .join(', ')}
        subtitle={subtitle}
        subtitleClassName={colorClassName ?? 'text-[color:var(--text-color-accent-blue)]'}
        description={description}
        tags={tags}
        selected={selected}
        className={emphasisClassName}
        onActivate={onActivate}
        stopPropagation={stopPropagation}
        leading={
          <CardIconBadge>
            <Icon size={16} className={iconClassName} role="img" aria-hidden />
          </CardIconBadge>
        }
      />
      {hasOutgoing && (
        <Handle type="source" position={sourcePosition} className="bg-strong border-none" />
      )}
    </>
  );
};
