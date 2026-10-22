// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Db, Beaker, ChatSingle, Rocket, Sparkle, Wand } from '@nvidia/react-gui-icons/icons';
import type { Meta, StoryObj } from '@storybook/react';
import { DagCanvas } from '@studio/components/DagCanvas';
import type { DagEdge, DagNode } from '@studio/components/DagCanvas/types';

const nodes: DagNode[] = [
  {
    id: 'ingest',
    data: {
      title: 'Ingest',
      type: 'SOURCE',
      description: 'Load raw data',
      icon: Db,
      status: 'success',
    },
  },
  {
    id: 'clean',
    data: {
      title: 'Clean',
      type: 'TRANSFORM',
      description: 'Dedupe & normalize',
      icon: Wand,
      status: 'success',
    },
  },
  {
    id: 'features',
    data: {
      title: 'Instruction',
      type: 'LLM TEXT',
      description: 'Writes a question about the topic',
      tags: ['{{topic}}', '{{difficulty}}'],
      icon: ChatSingle,
      status: 'running',
    },
  },
  {
    id: 'train',
    data: {
      title: 'Train',
      type: 'GENERATE',
      description: 'Fit model',
      icon: Sparkle,
      status: 'idle',
    },
  },
  {
    id: 'evaluate',
    data: {
      title: 'Evaluate',
      type: 'VALIDATE',
      description: 'Score on holdout',
      icon: Beaker,
      status: 'idle',
    },
  },
  {
    id: 'deploy',
    data: {
      title: 'Deploy',
      type: 'SINK',
      description: 'Ship to gateway',
      icon: Rocket,
      status: 'error',
      colorClassName: 'text-[color:var(--text-color-feedback-danger)]',
      tags: ['{{deployment_id}}', '{{model_id}}'],
    },
  },
];

const edges: DagEdge[] = [
  { source: 'ingest', target: 'clean' },
  { source: 'clean', target: 'features' },
  { source: 'features', target: 'train' },
  { source: 'train', target: 'evaluate' },
  { source: 'evaluate', target: 'deploy' },
  { source: 'features', target: 'evaluate', label: 'baseline' },
];

const meta = {
  component: DagCanvas,
  title: 'Components/DagCanvas',
  parameters: {
    layout: 'fullscreen',
  },
  args: {
    nodes,
    edges,
    direction: 'TB',
    // eslint-disable-next-line no-console
    onNodeClick: (id, data) => console.log('node click', id, data),
  },
  decorators: [
    (Story) => (
      <div className="h-dvh w-dvw bg-surface-base">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof DagCanvas>;

export default meta;
type Story = StoryObj<typeof meta>;

/** A six-step pipeline laid out top-to-bottom. */
export const Default: Story = {};

/** The same graph flowing left-to-right. */
export const LeftToRight: Story = {
  args: { direction: 'LR' },
};
