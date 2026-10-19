// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MIN_GRAPH_ZOOM, readViewport } from '@studio/components/DagCanvas/viewport';
import { useNodes, useNodesInitialized, useReactFlow } from '@xyflow/react';
import { type FC, useEffect, useRef } from 'react';

interface FitGraphControllerProps {
  readonly viewportStorageKey?: string;
  readonly selectedNodeId?: string | null;
}

export const FitGraphController: FC<FitGraphControllerProps> = ({
  viewportStorageKey,
  selectedNodeId,
}) => {
  const { fitView } = useReactFlow();
  const nodes = useNodes();
  const nodesInitialized = useNodesInitialized();
  const graphId = nodes.map((node) => node.id).join('|');
  const lastFittedGraph = useRef<string | null>(null);

  useEffect(() => {
    if (!nodesInitialized || !graphId) return;
    if (lastFittedGraph.current === null) {
      lastFittedGraph.current = graphId;
      return;
    }
    if (graphId === lastFittedGraph.current) return;
    lastFittedGraph.current = graphId;
    requestAnimationFrame(() => {
      const savedViewport = viewportStorageKey ? readViewport(viewportStorageKey) : null;
      const savedZoom = savedViewport
        ? Math.min(Math.max(savedViewport.zoom, MIN_GRAPH_ZOOM), 2)
        : null;
      const selectedNode = selectedNodeId && nodes.some(({ id }) => id === selectedNodeId);
      fitView({
        nodes: selectedNode ? [{ id: selectedNodeId }] : undefined,
        padding: 0.12,
        minZoom: savedZoom ?? MIN_GRAPH_ZOOM,
        maxZoom: savedZoom ?? 1,
      });
    });
  }, [fitView, graphId, nodes, nodesInitialized, selectedNodeId, viewportStorageKey]);

  return null;
};
