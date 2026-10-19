// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DagCanvas } from '@studio/components/DagCanvas';
import {
  CardNode,
  type CardNodeData,
  type CardNodeType,
} from '@studio/components/DagCanvas/CardNode';
import { NODE_HEIGHT, NODE_WIDTH, layoutGraph } from '@studio/components/DagCanvas/layout';
import { getGraphMotionDuration } from '@studio/components/DagCanvas/motion';
import type { DagNode } from '@studio/components/DagCanvas/types';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { type Edge, type Node, type NodeProps, Position } from '@xyflow/react';
import { Sparkles } from 'lucide-react';

// React Flow reads from an internal store/context that only exists inside a fully
// measured <ReactFlow> (needs ResizeObserver + real layout, absent in jsdom). Stub
// Handle for isolated CardNode tests, and stub ReactFlow to render each node's
// activation handler so DagCanvas's own onActivate → onNodeClick wiring is testable.
vi.mock('@xyflow/react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@xyflow/react')>();
  return {
    ...actual,
    Handle: () => null,
    Background: () => null,
    Controls: () => null,
    ReactFlow: ({
      nodes,
      edges,
      minZoom,
      nodesFocusable,
      edgesFocusable,
      onNodesChange,
      onNodesDelete,
    }: {
      nodes: CardNodeType[];
      edges: Edge[];
      minZoom?: number;
      nodesFocusable?: boolean;
      edgesFocusable?: boolean;
      onNodesChange?: (changes: Array<{ id: string; type: 'select'; selected: boolean }>) => void;
      onNodesDelete?: (nodes: CardNodeType[]) => void;
    }) => (
      <div
        data-testid="react-flow"
        data-min-zoom={minZoom}
        data-nodes-focusable={String(nodesFocusable)}
        data-edges-focusable={String(edgesFocusable)}
        data-nodes-deletable={String(nodes.every(({ deletable }) => deletable !== false))}
        data-edges-deletable={String(edges.every(({ deletable }) => deletable !== false))}
      >
        {nodes.map((node) => (
          <button
            key={node.id}
            type="button"
            aria-pressed={node.selected}
            onClick={() => node.data.onActivate?.()}
          >
            {node.data.title}
          </button>
        ))}
        <button
          type="button"
          onClick={() =>
            nodes[0] && onNodesChange?.([{ id: nodes[0].id, type: 'select', selected: true }])
          }
        >
          Select first node
        </button>
        <button
          type="button"
          onClick={() => onNodesDelete?.(nodes.filter(({ selected }) => selected))}
        >
          Delete selected nodes
        </button>
      </div>
    ),
  };
});

const makeNode = (id: string): Node<CardNodeData> => ({
  id,
  position: { x: 0, y: 0 },
  data: { title: id },
});

const mediaQuery = (matches: boolean): MediaQueryList =>
  ({
    matches,
    media: '(prefers-reduced-motion: reduce)',
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }) as MediaQueryList;

describe('layoutGraph', () => {
  it('assigns a distinct position to every node', () => {
    const nodes = [makeNode('a'), makeNode('b'), makeNode('c')];
    const edges: Edge[] = [
      { id: 'a-b', source: 'a', target: 'b' },
      { id: 'b-c', source: 'b', target: 'c' },
    ];

    const result = layoutGraph(nodes, edges, 'TB');

    expect(result).toHaveLength(3);
    const ys = result.map((node) => node.position.y);
    // Top-to-bottom: each rank sits below the previous one.
    expect(new Set(ys).size).toBe(3);
    expect(result.every((node) => Number.isFinite(node.position.x))).toBe(true);
  });

  it('orients handles top/bottom for TB and left/right for LR', () => {
    const nodes = [makeNode('a'), makeNode('b')];
    const edges: Edge[] = [{ id: 'a-b', source: 'a', target: 'b' }];

    const [tb] = layoutGraph(nodes, edges, 'TB');
    expect(tb.targetPosition).toBe(Position.Top);
    expect(tb.sourcePosition).toBe(Position.Bottom);

    const [lr] = layoutGraph(nodes, edges, 'LR');
    expect(lr.targetPosition).toBe(Position.Left);
    expect(lr.sourcePosition).toBe(Position.Right);
  });

  it('moves a node off a skip edge that would otherwise run through it', () => {
    // a → b → c chained, plus a → c skipping over b. dagre stacks a and c in the
    // same column, so the straight a→c edge (and its label) would cross b.
    const nodes = [makeNode('a'), makeNode('b'), makeNode('c')];
    const edges: Edge[] = [
      { id: 'a-b', source: 'a', target: 'b' },
      { id: 'b-c', source: 'b', target: 'c' },
      { id: 'a-c', source: 'a', target: 'c', label: 'skip' },
    ];

    const result = layoutGraph(nodes, edges, 'TB');
    const byId = Object.fromEntries(result.map((n) => [n.id, n]));

    // The skip edge is drawn straight down a and c's shared column.
    const column = byId.a.position.x;
    expect(byId.c.position.x).toBeCloseTo(column);

    // b's card must not straddle that column.
    const bLeft = byId.b.position.x;
    const bRight = byId.b.position.x + NODE_WIDTH;
    expect(column < bLeft || column > bRight).toBe(true);
  });

  it('offsets positions from dagre centers by half the card size', () => {
    const [only] = layoutGraph([makeNode('solo')], [], 'TB');
    // A single node centers at (NODE_WIDTH/2, NODE_HEIGHT/2), so the top-left origin is (0, 0).
    expect(only.position.x).toBeCloseTo(0);
    expect(only.position.y).toBeCloseTo(0);
    expect(NODE_WIDTH).toBeGreaterThan(0);
    expect(NODE_HEIGHT).toBeGreaterThan(0);
  });

  it('lays out the maximum Intake page without dropping nodes', () => {
    const nodes = Array.from({ length: 1000 }, (_, index) => makeNode(`node-${index}`));
    const edges = nodes.slice(1).map((node, index) => ({
      id: `edge-${index}`,
      source: nodes[index].id,
      target: node.id,
    }));

    const result = layoutGraph(nodes, edges, 'LR');

    expect(result).toHaveLength(1000);
    expect(result.every(({ position }) => Number.isFinite(position.x))).toBe(true);
    expect(result.every(({ position }) => Number.isFinite(position.y))).toBe(true);
  });
});

const renderCard = (data: CardNodeType['data']) =>
  render(<CardNode {...({ data } as unknown as NodeProps<CardNodeType>)} />);

describe('CardNode', () => {
  it('renders the title, type label, description, and tags', () => {
    renderCard({
      title: 'Instruction',
      type: 'LLM TEXT',
      description: 'Writes a question about the topic',
      tags: ['{{topic}}', '{{difficulty}}'],
    });
    expect(screen.getByText('Instruction')).toBeInTheDocument();
    expect(screen.getByText('LLM TEXT')).toBeInTheDocument();
    expect(screen.getByText('Writes a question about the topic')).toBeInTheDocument();
    expect(screen.getByText('{{topic}}')).toBeInTheDocument();
    expect(screen.getByText('{{difficulty}}')).toBeInTheDocument();
  });

  it('fires onActivate when clicked', async () => {
    const user = userEvent.setup();
    const onActivate = vi.fn();
    renderCard({ title: 'Deploy', onActivate });

    await user.click(screen.getByRole('button', { name: /Deploy/ }));

    expect(onActivate).toHaveBeenCalledTimes(1);
  });

  it('is keyboard-activatable', async () => {
    const user = userEvent.setup();
    const onActivate = vi.fn();
    renderCard({ title: 'Evaluate', onActivate });

    await user.tab();
    await user.keyboard('{Enter}');

    expect(onActivate).toHaveBeenCalled();
  });

  it('includes status in the accessible node name', () => {
    renderCard({ title: 'Retrieve context', type: 'TOOL', status: 'error' });

    expect(screen.getByRole('button', { name: 'Retrieve context, TOOL, error' })).toBeVisible();
    expect(screen.getByText('TOOL · error')).toBeVisible();
  });

  it('keeps the type accent on the icon when the status is not idle', () => {
    renderCard({
      title: 'Summarize',
      type: 'LLM',
      status: 'error',
      icon: Sparkles,
      colorClassName: 'text-[color:var(--text-color-accent-purple)]',
    });

    expect(screen.getByRole('img', { hidden: true })).toHaveClass(
      'text-[color:var(--text-color-accent-purple)]'
    );
  });

  it('announces longest-path membership and keeps selection visually distinct', () => {
    const { rerender } = render(
      <CardNode
        {...({
          data: { title: 'Retrieve context', highlighted: true },
          selected: false,
        } as unknown as NodeProps<CardNodeType>)}
      />
    );

    const highlighted = screen.getByRole('button', {
      name: 'Retrieve context, longest path',
    });
    expect(highlighted).toHaveClass('!border-2', '!border-dashed', '!border-brand');

    rerender(
      <CardNode
        {...({
          data: { title: 'Retrieve context', highlighted: true },
          selected: true,
        } as unknown as NodeProps<CardNodeType>)}
      />
    );
    expect(screen.getByRole('button', { name: 'Retrieve context, longest path' })).toHaveClass(
      'border-brand'
    );
    expect(screen.getByRole('button', { name: 'Retrieve context, longest path' })).not.toHaveClass(
      '!border-dashed'
    );
  });
});

describe('DagCanvas', () => {
  it('disables camera animation when reduced motion is requested', () => {
    const matchMedia = vi.spyOn(window, 'matchMedia');
    matchMedia.mockReturnValue(mediaQuery(true));
    expect(getGraphMotionDuration()).toBe(0);

    matchMedia.mockReturnValue(mediaQuery(false));
    expect(getGraphMotionDuration()).toBe(500);
  });

  it('uses the card button as the only node focus target', () => {
    render(<DagCanvas nodes={[{ id: 'train', data: { title: 'Train' } }]} edges={[]} />);

    expect(screen.getByTestId('react-flow')).toHaveAttribute('data-nodes-focusable', 'false');
    expect(screen.getByTestId('react-flow')).toHaveAttribute('data-edges-focusable', 'false');
    expect(screen.getAllByRole('button', { name: 'Train' })).toHaveLength(1);
  });

  it('bridges a node activation to onNodeClick with the node id and data', async () => {
    const user = userEvent.setup();
    const onNodeClick = vi.fn();
    const nodes: DagNode[] = [
      { id: 'train', data: { title: 'Train', type: 'CUSTOMIZER' } },
      { id: 'evaluate', data: { title: 'Evaluate' } },
    ];

    render(
      <DagCanvas
        nodes={nodes}
        edges={[{ source: 'train', target: 'evaluate' }]}
        onNodeClick={onNodeClick}
      />
    );

    await user.click(screen.getByRole('button', { name: 'Train' }));

    expect(onNodeClick).toHaveBeenCalledTimes(1);
    expect(onNodeClick).toHaveBeenCalledWith('train', nodes[0].data);
  });

  it('always calls the latest onNodeClick after the callback identity changes', async () => {
    const user = userEvent.setup();
    const first = vi.fn();
    const second = vi.fn();
    const nodes: DagNode[] = [{ id: 'train', data: { title: 'Train' } }];

    const { rerender } = render(<DagCanvas nodes={nodes} edges={[]} onNodeClick={first} />);
    // Swap in a fresh callback reference, mirroring a parent passing an inline arrow.
    rerender(<DagCanvas nodes={nodes} edges={[]} onNodeClick={second} />);

    await user.click(screen.getByRole('button', { name: 'Train' }));

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledWith('train', nodes[0].data);
  });

  it('marks the selected node without requiring React Flow selection state', () => {
    const nodes: DagNode[] = [
      { id: 'train', data: { title: 'Train' } },
      { id: 'evaluate', data: { title: 'Evaluate' } },
    ];

    render(<DagCanvas nodes={nodes} edges={[]} selectedNodeId="evaluate" />);

    expect(screen.getByRole('button', { name: 'Evaluate' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(screen.getByRole('button', { name: 'Train' })).toHaveAttribute('aria-pressed', 'false');
  });

  it('uses a zoom range that can contain the maximum trace page', () => {
    render(<DagCanvas nodes={[{ id: 'root', data: { title: 'Root' } }]} edges={[]} />);

    expect(screen.getByTestId('react-flow')).toHaveAttribute('data-min-zoom', '0.001');
  });

  it('keeps non-editable graphs from deleting nodes or edges', () => {
    render(
      <DagCanvas
        nodes={[
          { id: 'root', data: { title: 'Root' } },
          { id: 'child', data: { title: 'Child' } },
        ]}
        edges={[{ source: 'root', target: 'child' }]}
      />
    );

    expect(screen.getByTestId('react-flow')).toHaveAttribute('data-nodes-deletable', 'false');
    expect(screen.getByTestId('react-flow')).toHaveAttribute('data-edges-deletable', 'false');
  });

  it('preserves React Flow selection and deletion for editable canvases', async () => {
    const user = userEvent.setup();
    const onNodeDelete = vi.fn();
    render(
      <DagCanvas
        nodes={[{ id: 'train', data: { title: 'Train' } }]}
        edges={[]}
        onNodeDelete={onNodeDelete}
      />
    );

    expect(screen.getByTestId('react-flow')).toHaveAttribute('data-nodes-focusable', 'true');
    await user.click(screen.getByRole('button', { name: 'Select first node' }));
    expect(screen.getByRole('button', { name: 'Train' })).toHaveAttribute('aria-pressed', 'true');

    await user.click(screen.getByRole('button', { name: 'Delete selected nodes' }));
    expect(onNodeDelete).toHaveBeenCalledWith('train');
  });
});
