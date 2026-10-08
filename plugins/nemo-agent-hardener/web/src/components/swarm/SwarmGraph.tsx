// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  EDGES,
  GROUP_COLOR,
  NODES,
  type NodeStatus,
  type SwarmNode,
  type SwarmState,
} from '@agent-hardener/components/swarm/swarmModel';
import { Button } from '@nvidia/foundations-react-core';
import { Maximize2, Minus, Plus } from 'lucide-react';
import { FC, PointerEvent as ReactPointerEvent, useRef, useState } from 'react';

interface SwarmGraphProps {
  swarm: SwarmState;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

interface Point {
  x: number;
  y: number;
}

const BASE_W = 1000;
const BASE_H = 720;
const MIN_ZOOM = 0.6;
const MAX_ZOOM = 3;
const DRAG_THRESHOLD_PX = 4; // movement below this on a node is a click (select), not a reposition
const clampZoom = (z: number): number => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));

// Labeled group regions drawn behind the nodes (the "swarm" framing from the demo).
const REGIONS = [
  { label: 'ATTACKER SWARM', x: 40, y: 90, w: 320, h: 300, color: GROUP_COLOR.attacker },
  { label: 'OPENSHELL SANDBOX', x: 400, y: 200, w: 200, h: 220, color: GROUP_COLOR.victim },
  { label: 'DEFENDER SWARM', x: 640, y: 90, w: 320, h: 300, color: GROUP_COLOR.defender },
  { label: 'VALIDATOR SWARM', x: 250, y: 540, w: 400, h: 160, color: GROUP_COLOR.validator },
];

const nodeRadius = (n: SwarmNode): number => (n.group === 'victim' ? 40 : n.isManager ? 32 : 26);

const fillFor = (color: string, status: NodeStatus): string => {
  if (status === 'failed') return 'rgba(255,56,85,0.18)';
  if (status === 'blocked') return 'rgba(255,171,64,0.18)';
  if (status === 'running') return `${color}33`;
  if (status === 'success') return `${color}44`;
  return 'rgba(255,255,255,0.03)';
};

const strokeFor = (color: string, status: NodeStatus): string => {
  if (status === 'failed') return '#ff3855';
  if (status === 'blocked') return '#ffab40';
  return color;
};

export const SwarmGraph: FC<SwarmGraphProps> = ({ swarm, selectedId, onSelect }) => {
  const [zoom, setZoom] = useState(1);
  const [center, setCenter] = useState({ x: BASE_W / 2, y: BASE_H / 2 });
  // Per-node position overrides so the user can restyle the layout (ephemeral; "Reset view" clears them).
  const [positions, setPositions] = useState<Record<string, Point>>({});
  const svgRef = useRef<SVGSVGElement>(null);
  const panRef = useRef<{ x: number; y: number } | null>(null);
  const nodeDragRef = useRef<{ id: string; base: Point; startX: number; startY: number } | null>(
    null
  );

  const vw = BASE_W / zoom;
  const vh = BASE_H / zoom;
  const viewBox = `${center.x - vw / 2} ${center.y - vh / 2} ${vw} ${vh}`;
  const posOf = (n: SwarmNode): Point => positions[n.id] ?? { x: n.x, y: n.y };
  // Client-pixel delta → SVG-unit delta under the current zoom.
  const toSvgDelta = (dxClient: number, dyClient: number, rect: DOMRect): Point => ({
    x: (dxClient * vw) / rect.width,
    y: (dyClient * vh) / rect.height,
  });

  const zoomBy = (factor: number) => setZoom((z) => clampZoom(z * factor));
  const reset = () => {
    setZoom(1);
    setCenter({ x: BASE_W / 2, y: BASE_H / 2 });
    setPositions({});
  };

  // A node pointer-down starts a node drag (and suppresses canvas pan); bare canvas pointer-down pans.
  const onNodePointerDown = (e: ReactPointerEvent<SVGGElement>, node: SwarmNode) => {
    e.stopPropagation();
    nodeDragRef.current = { id: node.id, base: posOf(node), startX: e.clientX, startY: e.clientY };
  };
  const onCanvasPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    panRef.current = { x: e.clientX, y: e.clientY };
  };
  const onPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const nd = nodeDragRef.current;
    if (nd) {
      const d = toSvgDelta(e.clientX - nd.startX, e.clientY - nd.startY, rect);
      setPositions((prev) => ({ ...prev, [nd.id]: { x: nd.base.x + d.x, y: nd.base.y + d.y } }));
      return;
    }
    const pan = panRef.current;
    if (!pan) return;
    const d = toSvgDelta(e.clientX - pan.x, e.clientY - pan.y, rect);
    panRef.current = { x: e.clientX, y: e.clientY };
    setCenter((c) => ({ x: c.x - d.x, y: c.y - d.y }));
  };
  const onPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    const nd = nodeDragRef.current;
    if (nd) {
      const moved = Math.hypot(e.clientX - nd.startX, e.clientY - nd.startY) > DRAG_THRESHOLD_PX;
      if (!moved) onSelect(nd.id);
      nodeDragRef.current = null;
    }
    panRef.current = null;
  };

  return (
    <div className="relative h-full w-full">
      <svg
        ref={svgRef}
        viewBox={viewBox}
        width="100%"
        height="100%"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Agent Hardener agents"
        className="cursor-grab touch-none active:cursor-grabbing"
        onPointerDown={onCanvasPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        {REGIONS.map((r) => (
          <g key={r.label}>
            <rect
              x={r.x}
              y={r.y}
              width={r.w}
              height={r.h}
              rx={16}
              fill={`${r.color}0d`}
              stroke={`${r.color}44`}
              strokeDasharray="6 6"
            />
            <text
              x={r.x + 12}
              y={r.y + 22}
              fill={r.color}
              fontSize={13}
              fontFamily="monospace"
              letterSpacing={1.5}
            >
              {r.label}
            </text>
          </g>
        ))}

        {EDGES.map((edge, i) => {
          const a = NODES.find((n) => n.id === edge.from);
          const b = NODES.find((n) => n.id === edge.to);
          if (!a || !b) return null;
          const pa = posOf(a);
          const pb = posOf(b);
          const active = swarm.statuses[edge.to] === 'running';
          const pathId = `iron-edge-${i}`;
          const d = `M ${pa.x} ${pa.y} L ${pb.x} ${pb.y}`;
          return (
            <g key={pathId}>
              <path
                id={pathId}
                d={d}
                fill="none"
                stroke={active ? GROUP_COLOR.victim : 'rgba(255,255,255,0.12)'}
                strokeWidth={active ? 2 : 1}
              />
              {active && (
                <circle r={4} fill={GROUP_COLOR.victim}>
                  <animateMotion dur="1.6s" repeatCount="indefinite">
                    <mpath href={`#${pathId}`} />
                  </animateMotion>
                </circle>
              )}
            </g>
          );
        })}

        {NODES.map((n) => {
          const status = swarm.statuses[n.id] ?? 'pending';
          const color = GROUP_COLOR[n.group];
          const r = nodeRadius(n);
          const selected = selectedId === n.id;
          const p = posOf(n);
          const count = n.isManager ? 0 : (swarm.nodeExchanges[n.id]?.length ?? 0);
          return (
            <g
              key={n.id}
              onPointerDown={(e) => onNodePointerDown(e, n)}
              className="cursor-grab active:cursor-grabbing"
            >
              {status === 'running' && (
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={r}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.5}
                  opacity={0.6}
                >
                  <animate
                    attributeName="r"
                    values={`${r};${r + 12}`}
                    dur="1.4s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="opacity"
                    values="0.6;0"
                    dur="1.4s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
              <circle
                cx={p.x}
                cy={p.y}
                r={r}
                fill={fillFor(color, status)}
                stroke={strokeFor(color, status)}
                strokeWidth={selected ? 4 : 2}
                opacity={status === 'pending' ? 0.5 : 1}
              />
              <text x={p.x} y={p.y + r + 16} textAnchor="middle" fill="#c9d6de" fontSize={12}>
                {n.title}
              </text>
              {n.isManager && (
                <text
                  x={p.x}
                  y={p.y - r - 8}
                  textAnchor="middle"
                  fill={color}
                  fontSize={9}
                  fontFamily="monospace"
                  letterSpacing={1}
                >
                  MANAGER
                </text>
              )}
              {count > 0 && (
                <g>
                  <circle cx={p.x + r * 0.72} cy={p.y - r * 0.72} r={9} fill={color} />
                  <text
                    x={p.x + r * 0.72}
                    y={p.y - r * 0.72 + 3}
                    textAnchor="middle"
                    fill="#0b0f14"
                    fontSize={9}
                    fontFamily="monospace"
                  >
                    {count}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
      <div className="absolute right-1 top-1 flex flex-col gap-1">
        <Button kind="secondary" size="small" aria-label="Zoom in" onClick={() => zoomBy(1.2)}>
          <Plus className="h-4 w-4" />
        </Button>
        <Button kind="secondary" size="small" aria-label="Zoom out" onClick={() => zoomBy(1 / 1.2)}>
          <Minus className="h-4 w-4" />
        </Button>
        <Button kind="secondary" size="small" aria-label="Reset view" onClick={reset}>
          <Maximize2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
