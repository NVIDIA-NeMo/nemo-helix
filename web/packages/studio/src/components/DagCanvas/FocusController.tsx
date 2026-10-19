// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getGraphMotionDuration } from '@studio/components/DagCanvas/motion';
import { useNodes, useReactFlow } from '@xyflow/react';
import { type FC, useEffect, useRef } from 'react';

/**
 * Pans/zooms the viewport to center `focusNodeId` whenever it changes to a node present
 * on the canvas. Rendered inside `ReactFlow` so it can use the flow hooks. Tracks the
 * last-focused id so live edits to a focused node don't re-center it, and defers focus
 * until a just-added node has actually landed in the store.
 */
interface FocusControllerProps {
  readonly focusNodeId?: string | null;
}

export const FocusController: FC<FocusControllerProps> = ({ focusNodeId }) => {
  const { fitView } = useReactFlow();
  const nodes = useNodes();
  const lastFocused = useRef<string | null>(null);

  useEffect(() => {
    if (!focusNodeId || focusNodeId === lastFocused.current) return;
    if (!nodes.some((node) => node.id === focusNodeId)) return;
    lastFocused.current = focusNodeId;
    fitView({
      nodes: [{ id: focusNodeId }],
      duration: getGraphMotionDuration(),
      padding: 0.4,
      maxZoom: 1.2,
    });
  }, [focusNodeId, nodes, fitView]);

  return null;
};
