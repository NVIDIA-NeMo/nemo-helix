// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CardNode, type CardNodeType } from '@studio/components/DagCanvas/CardNode';
import { CenterNodeController } from '@studio/components/DagCanvas/CenterNodeController';
import { FitGraphController } from '@studio/components/DagCanvas/FitGraphController';
import { FitNodesController } from '@studio/components/DagCanvas/FitNodesController';
import { FocusController } from '@studio/components/DagCanvas/FocusController';
import { layoutGraph } from '@studio/components/DagCanvas/layout';
import type {
  DagDirection,
  DagEdge,
  DagNode,
  DagNodeData,
} from '@studio/components/DagCanvas/types';
import { useNvColorMode } from '@studio/components/DagCanvas/useNvColorMode';
import { MIN_GRAPH_ZOOM, readViewport } from '@studio/components/DagCanvas/viewport';
import {
  Background,
  type ColorMode,
  Controls,
  type Edge,
  MarkerType,
  type NodeChange,
  ReactFlow,
  type Viewport,
  useEdgesState,
  useNodesState,
} from '@xyflow/react';
import '@studio/components/DagCanvas/DagCanvas.css';
import '@xyflow/react/dist/style.css';
import { type FC, useCallback, useEffect, useMemo, useRef } from 'react';

const NODE_TYPES = { card: CardNode };

/** Stable id for an edge; falls back to a source/target pair when none is given. */
const edgeId = (edge: DagEdge): string => edge.id ?? `${edge.source}->${edge.target}`;

export interface DagCanvasProps {
  /** Graph nodes; positions are computed automatically. */
  nodes: DagNode[];
  edges: DagEdge[];
  onNodeClick?: (id: string, data: DagNodeData) => void;
  /** Node shown as selected without moving the viewport. */
  selectedNodeId?: string | null;
  /** When set (or changed), the viewport animates to center this node. */
  focusNodeId?: string | null;
  /** When set, the viewport animates to include all listed nodes. */
  fitNodeIds?: readonly string[];
  /** When requested, the viewport centers this node without changing zoom. */
  centerNodeId?: string | null;
  centerNodeNonce?: number;
  /** Fired for each node removed via the canvas (e.g. Backspace on a selected node). */
  onNodeDelete?: (id: string) => void;
  /** Layout flow direction; defaults to `'TB'` (top-to-bottom). */
  direction?: DagDirection;
  /**
   * Light/dark mode for the canvas. Defaults to following the Studio theme (the
   * `nv-dark` class on `<html>`). Set explicitly to override.
   */
  colorMode?: ColorMode;
  /** Session storage key used to retain the user's viewport when the canvas remounts. */
  viewportStorageKey?: string;
  className?: string;
}

/** The host element must have a defined size (e.g. `h-full w-full` inside a sized parent); React Flow fills its container. */
export const DagCanvas: FC<DagCanvasProps> = ({
  nodes,
  edges,
  onNodeClick,
  selectedNodeId,
  focusNodeId,
  fitNodeIds,
  centerNodeId,
  centerNodeNonce,
  onNodeDelete,
  direction = 'TB',
  colorMode,
  viewportStorageKey,
  className,
}) => {
  const themeColorMode = useNvColorMode();
  const initialViewport = useMemo(
    () => (viewportStorageKey ? readViewport(viewportStorageKey) : null),
    [viewportStorageKey]
  );
  const onNodeClickRef = useRef(onNodeClick);
  useEffect(() => {
    onNodeClickRef.current = onNodeClick;
  }, [onNodeClick]);

  const laidOutNodes = useMemo<CardNodeType[]>(() => {
    // A node only shows the handle for a side that actually has an edge, so
    // fully-unconnected nodes render no handles.
    const targets = new Set(edges.map((edge) => edge.target));
    const sources = new Set(edges.map((edge) => edge.source));
    const rfNodes: CardNodeType[] = nodes.map((node) => ({
      id: node.id,
      type: 'card',
      position: { x: 0, y: 0 },
      selected: false,
      data: {
        ...node.data,
        onActivate: () => onNodeClickRef.current?.(node.id, node.data),
        hasIncoming: targets.has(node.id),
        hasOutgoing: sources.has(node.id),
        stopPropagation: !onNodeDelete,
      },
    }));
    const rfEdges: Edge[] = edges.map((edge) => ({
      id: edgeId(edge),
      source: edge.source,
      target: edge.target,
      // Pass the label through so dagre can reserve space for it during layout.
      label: edge.label,
    }));
    return layoutGraph(rfNodes, rfEdges, direction);
  }, [edges, nodes, direction, onNodeDelete]);

  const renderedNodes = useMemo<CardNodeType[]>(
    () =>
      laidOutNodes.map((node) => ({
        ...node,
        deletable: Boolean(onNodeDelete),
        selected: node.id === selectedNodeId,
      })),
    [laidOutNodes, onNodeDelete, selectedNodeId]
  );

  const styledEdges = useMemo<Edge[]>(
    () =>
      edges.map((edge) => ({
        id: edgeId(edge),
        source: edge.source,
        target: edge.target,
        label: edge.label,
        type: 'smoothstep',
        markerEnd: {
          type: MarkerType.ArrowClosed,
          ...(edge.highlighted ? { color: 'var(--border-color-brand)' } : undefined),
        },
        className: edge.highlighted ? 'nemo-dag-path-edge' : undefined,
        deletable: false,
        animated: false,
        style: edge.highlighted
          ? { stroke: 'var(--border-color-brand)', strokeWidth: 2.5 }
          : edge.muted
            ? { opacity: 0.2 }
            : undefined,
      })),
    [edges]
  );

  const [flowNodes, setFlowNodes, onNodesChange] = useNodesState(renderedNodes);
  const [flowEdges, setFlowEdges, onEdgesChange] = useEdgesState(styledEdges);

  // Re-sync internal React Flow state when the input graph (or layout) changes.
  useEffect(() => setFlowNodes(renderedNodes), [renderedNodes, setFlowNodes]);
  useEffect(() => setFlowEdges(styledEdges), [styledEdges, setFlowEdges]);
  const handleMoveEnd = useCallback(
    (_: MouseEvent | TouchEvent | null, viewport: Viewport) => {
      if (!viewportStorageKey) return;
      try {
        sessionStorage.setItem(viewportStorageKey, JSON.stringify(viewport));
      } catch {
        // Storage can be unavailable in privacy-restricted browser contexts.
      }
    },
    [viewportStorageKey]
  );
  const handleNodesChange = useCallback(
    (changes: NodeChange<CardNodeType>[]) =>
      onNodesChange(onNodeDelete ? changes : changes.filter((change) => change.type !== 'select')),
    [onNodeDelete, onNodesChange]
  );

  return (
    <div className={`size-full bg-surface-sunken ${className ?? ''}`}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={NODE_TYPES}
        colorMode={colorMode ?? themeColorMode}
        onNodesChange={handleNodesChange}
        onEdgesChange={onEdgesChange}
        onMoveEnd={handleMoveEnd}
        onNodesDelete={(deleted) => deleted.forEach((node) => onNodeDelete?.(node.id))}
        nodesDraggable={false}
        nodesFocusable={Boolean(onNodeDelete)}
        edgesFocusable={false}
        elevateNodesOnSelect={false}
        fitViewOptions={{ padding: 0.12, minZoom: MIN_GRAPH_ZOOM, maxZoom: 1 }}
        minZoom={MIN_GRAPH_ZOOM}
        maxZoom={2}
        onlyRenderVisibleElements={flowNodes.length > 250}
        defaultViewport={initialViewport ?? undefined}
        fitView={!initialViewport}
        proOptions={{ hideAttribution: true }}
      >
        <Background />
        <Controls position="top-left" />
        <FitGraphController
          viewportStorageKey={viewportStorageKey}
          selectedNodeId={selectedNodeId}
        />
        <FocusController focusNodeId={focusNodeId} />
        <FitNodesController nodeIds={fitNodeIds} />
        <CenterNodeController centerNodeId={centerNodeId} requestNonce={centerNodeNonce} />
      </ReactFlow>
    </div>
  );
};
