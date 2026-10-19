// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@nvidia/foundations-react-core';
import { DagCanvas } from '@studio/components/DagCanvas';
import { buildGraph } from '@studio/routes/DataDesignerJobBuildRoute/columns';
import type { JobBuilderFormValues } from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';
import { type FC, memo, useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export interface BuilderCanvasProps {
  focusNodeId: string | null;
  onNodeClick: (id: string | null) => void;
  onNodeDelete: (id: string) => void;
}

/** The graph is the sole subscriber for prompt/reference edits in canvas view. */
export const BuilderCanvas: FC<BuilderCanvasProps> = memo(function BuilderCanvas({
  focusNodeId,
  onNodeClick,
  onNodeDelete,
}) {
  const { control } = useFormContext<JobBuilderFormValues>();
  const columnRecord = useWatch({ control, name: 'columns' });
  const { nodes, edges } = useMemo(() => buildGraph(columnRecord), [columnRecord]);

  if (nodes.length === 0) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Text kind="body/regular/md" className="text-secondary">
          Empty canvas — add a column from the left to get started.
        </Text>
      </Flex>
    );
  }

  return (
    <DagCanvas
      nodes={nodes}
      edges={edges}
      onNodeClick={onNodeClick}
      onNodeDelete={onNodeDelete}
      focusNodeId={focusNodeId}
    />
  );
});
