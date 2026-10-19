// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Grid } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { DashboardPanel } from '@studio/routes/WorkspaceDashboardRoute/DashboardPanel';
import { Bot, Gauge, Settings, Upload } from 'lucide-react';
import type { ComponentType } from 'react';

const meta = {
  component: DashboardPanel,
  title: 'Routes/WorkspaceDashboardRoute/DashboardPanel',
} satisfies Meta<typeof DashboardPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

const cardDecorator = [
  (Story: ComponentType) => (
    <div className="w-[386px]">
      <Story />
    </div>
  ),
];

export const Default: Story = {
  decorators: cardDecorator,
  args: {
    icon: <Bot className="w-6 h-6" />,
    title: 'Connect an Agent',
    description: 'Register an agent you already run and route it through the gateway.',
    actions: [
      { icon: <Upload size={14} />, label: 'Upload an Agent', href: '#' },
      { icon: <Gauge size={14} />, label: 'Evaluate Performance', href: '#' },
      { icon: <Settings size={14} />, label: 'Optimize', href: '#' },
    ],
  },
};

export const SingleAction: Story = {
  decorators: cardDecorator,
  args: {
    icon: <Bot className="w-6 h-6" />,
    title: 'Chat with a Model',
    description: 'Chat with base models and explore capabilities.',
    actions: [{ icon: <Settings size={14} />, label: 'Chat', href: '#' }],
  },
};

export const ManyActions: Story = {
  decorators: cardDecorator,
  args: {
    icon: <Bot className="w-6 h-6" />,
    title: 'Automate a Workflow',
    description: 'An arbitrary number of quick actions can be attached to a card.',
    actions: [
      { icon: <Upload size={14} />, label: 'Upload an Agent', href: '#' },
      { icon: <Gauge size={14} />, label: 'Evaluate Performance', href: '#' },
      { icon: <Settings size={14} />, label: 'Optimize', href: '#' },
      { icon: <Bot size={14} />, label: 'Deploy', href: '#' },
      { icon: <Upload size={14} />, label: 'Publish', href: '#' },
    ],
  },
};

export const LongContent: Story = {
  decorators: cardDecorator,
  args: {
    icon: <Bot className="w-6 h-6" />,
    title: 'An extremely long dashboard card title that should wrap',
    description:
      'This is a very long description that tests how the card handles overflow across multiple lines while keeping the actions aligned to the bottom.',
    actions: [
      { icon: <Upload size={14} />, label: 'Upload an Agent', href: '#' },
      { icon: <Gauge size={14} />, label: 'Evaluate Performance', href: '#' },
      { icon: <Settings size={14} />, label: 'Optimize', href: '#' },
    ],
  },
};

/**
 * Demonstrates that cards with differing content lengths stay the same height
 * when laid out together in a grid (no stair-stepping).
 */
export const EqualHeightGrid: Story = {
  args: Default.args,
  render: (args) => (
    <Grid cols={{ md: 1, lg: 3 }} gap="density-xl" className="w-[1200px]">
      <DashboardPanel {...args} />
      <DashboardPanel {...SingleAction.args} />
      <DashboardPanel {...LongContent.args} />
    </Grid>
  ),
};
