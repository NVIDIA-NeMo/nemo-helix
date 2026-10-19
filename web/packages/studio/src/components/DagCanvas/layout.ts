// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import dagre from '@dagrejs/dagre';
import type { DagDirection } from '@studio/components/DagCanvas/types';
import { type Edge, type Node, Position } from '@xyflow/react';

/** Card dimensions fed to dagre so it can reserve space and route arrows. */
export const NODE_WIDTH = 240;
export const NODE_HEIGHT = 116;

/** Minimum empty gap to keep between a node and an edge that merely passes by it. */
const EDGE_CLEARANCE = 24;
const MAX_EDGE_CLEARANCE_NODES = 250;

/** Rough on-screen size of an edge label, used to keep nodes clear of it. */
const LABEL_CHAR_WIDTH = 7;
const LABEL_PADDING = 16;
const LABEL_HEIGHT = 20;
const estimateLabelWidth = (label: string): number =>
  label.length * LABEL_CHAR_WIDTH + LABEL_PADDING;

interface Point {
  x: number;
  y: number;
}

/**
 * Nudges nodes out from under edges that skip over their rank.
 *
 * React Flow draws an edge as a straight run between the source and target
 * handles, ignoring dagre's routing. When a "skip" edge connects two nodes that
 * dagre stacks in the same column (e.g. `features → evaluate`), that straight run
 * — and the label React Flow centers on it — passes right through whatever node
 * sits between them (`train`). dagre nudges the middle node aside but not always
 * far enough, and never accounts for the label's width.
 *
 * So after layout we shift any intervening node along the cross-axis until the
 * edge's column (plus its label and a clearance gap) no longer touches it. Mutates
 * the passed centers in place.
 */
const clearNodesOffEdges = (
  centers: Map<string, Point>,
  edges: Edge[],
  isHorizontal: boolean
): void => {
  edges.forEach((edge) => {
    const source = centers.get(edge.source);
    const target = centers.get(edge.target);
    if (!source || !target) return;

    // Cross-axis half-width to keep clear: half the node, the clearance gap, and
    // half the label (label width along the flow's cross-axis).
    const label = typeof edge.label === 'string' ? edge.label : undefined;
    const labelHalf = label ? (isHorizontal ? LABEL_HEIGHT : estimateLabelWidth(label)) / 2 : 0;
    const nodeHalf = (isHorizontal ? NODE_HEIGHT : NODE_WIDTH) / 2;
    const keepClear = nodeHalf + EDGE_CLEARANCE + labelHalf;

    // Along-flow span the edge covers (y for top-down, x for left-right).
    const start = isHorizontal ? source.x : source.y;
    const end = isHorizontal ? target.x : target.y;
    const [lo, hi] = start < end ? [start, end] : [end, start];

    centers.forEach((center, id) => {
      if (id === edge.source || id === edge.target) return;

      const along = isHorizontal ? center.x : center.y;
      // Only nodes whose rank falls strictly between the edge's endpoints.
      if (along <= lo + 1 || along >= hi - 1) return;

      // Where the straight edge sits on the cross-axis at this node's rank.
      const t = (along - start) / (end - start);
      const edgeCross = isHorizontal
        ? source.y + (target.y - source.y) * t
        : source.x + (target.x - source.x) * t;

      const cross = isHorizontal ? center.y : center.x;
      const delta = cross - edgeCross;
      if (Math.abs(delta) >= keepClear) return;

      const direction = delta === 0 ? 1 : Math.sign(delta);
      const shifted = edgeCross + direction * keepClear;
      if (isHorizontal) center.y = shifted;
      else center.x = shifted;
    });
  });
};

/**
 * Runs a top-down (or left-right) DAG layout over `nodes`/`edges` with dagre and
 * returns React Flow nodes with absolute `position` set, plus the correct
 * source/target handle positions for the chosen direction.
 *
 * dagre reports node centers, so we offset by half the card size to get the
 * top-left origin React Flow expects.
 */
export const layoutGraph = <N extends Node>(
  nodes: N[],
  edges: Edge[],
  direction: DagDirection
): N[] => {
  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({ rankdir: direction, nodesep: 48, ranksep: 64 });

  nodes.forEach((node) => {
    graph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });
  edges.forEach((edge) => {
    graph.setEdge(edge.source, edge.target);
  });

  dagre.layout(graph);

  const isHorizontal = direction === 'LR';

  const centers = new Map<string, Point>(
    nodes.map((node) => {
      const { x, y } = graph.node(node.id);
      return [node.id, { x, y }];
    })
  );
  if (nodes.length <= MAX_EDGE_CLEARANCE_NODES) {
    clearNodesOffEdges(centers, edges, isHorizontal);
  }

  return nodes.map((node) => {
    const { x, y } = centers.get(node.id) ?? { x: 0, y: 0 };
    return {
      ...node,
      targetPosition: isHorizontal ? Position.Left : Position.Top,
      sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
      position: { x: x - NODE_WIDTH / 2, y: y - NODE_HEIGHT / 2 },
    };
  });
};
