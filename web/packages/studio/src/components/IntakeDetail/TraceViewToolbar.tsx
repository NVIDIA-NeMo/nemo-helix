// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, SegmentedControl } from '@nvidia/foundations-react-core';
import { ChevronsDownUp, ChevronsUpDown } from 'lucide-react';
import type { FC } from 'react';

export type TraceViewMode = 'tree' | 'graph' | 'list';

interface TraceViewToolbarProps {
  viewMode: TraceViewMode;
  onViewModeChange: (viewMode: TraceViewMode) => void;
  showGraph?: boolean;
  onCollapseAll?: () => void;
  onExpandAll?: () => void;
}

/** Shared trace view toolbar for session and trace-selected detail bodies. */
export const TraceViewToolbar: FC<TraceViewToolbarProps> = ({
  viewMode,
  onViewModeChange,
  showGraph = true,
  onCollapseAll,
  onExpandAll,
}) => (
  <Flex align="center" justify="between" gap="density-lg" className="min-w-0">
    <SegmentedControl
      size="tiny"
      value={viewMode}
      onValueChange={(value) => onViewModeChange(value as TraceViewMode)}
      items={[
        { value: 'tree', children: 'Tree' },
        ...(showGraph ? [{ value: 'graph', children: 'Graph' }] : []),
        { value: 'list', children: 'List' },
      ]}
    />
    {onCollapseAll && onExpandAll ? (
      <Flex align="center" gap="density-xs">
        <Button
          kind="tertiary"
          size="tiny"
          type="button"
          aria-label="Collapse all"
          title="Collapse all"
          onClick={onCollapseAll}
        >
          <ChevronsDownUp size={14} aria-hidden />
        </Button>
        <Button
          kind="tertiary"
          size="tiny"
          type="button"
          aria-label="Expand all"
          title="Expand all"
          onClick={onExpandAll}
        >
          <ChevronsUpDown size={14} aria-hidden />
        </Button>
      </Flex>
    ) : null}
  </Flex>
);
