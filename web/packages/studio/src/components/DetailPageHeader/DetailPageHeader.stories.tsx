// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Button, TabsRoot } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import { DetailPageHeader } from '@studio/components/DetailPageHeader';
import { useState } from 'react';

const meta: Meta<typeof DetailPageHeader> = {
  component: DetailPageHeader,
  title: 'Components/DetailPageHeader',
  decorators: [
    (Story, { args }) => {
      const [tab, setTab] = useState(args.tabs[0]?.value);
      return (
        <TabsRoot value={tab} onValueChange={setTab}>
          <div className="p-4">
            <Story />
          </div>
        </TabsRoot>
      );
    },
  ],
};

export default meta;

type Story = StoryObj<typeof DetailPageHeader>;

const modelTabs = [
  { value: 'details', label: 'Details' },
  { value: 'chat', label: 'Chat' },
  { value: 'training', label: 'Training' },
  { value: 'logs', label: 'Logs' },
];

const tags = (...labels: string[]) =>
  labels.map((label) => (
    <Badge key={label} kind="outline" color="gray">
      {label}
    </Badge>
  ));

export const Default: Story = {
  args: {
    heading: 'llama-3.2-1b-support',
    slotStatus: (
      <DeploymentStatusBadge state={{ kind: 'served', hasDeployment: true }} isAdapter={false} />
    ),
    slotTags: tags('Full SFT', 'Base: meta/llama-3.2-1b-instruct'),
    description: 'Fine-tuned on support tickets to answer billing questions.',
    tabs: modelTabs,
  },
};

export const Adapter: Story = {
  args: {
    heading: 'support-lora',
    slotStatus: <DeploymentStatusBadge state={{ kind: 'adapter-not-loaded' }} isAdapter />,
    slotTags: tags('LoRA', 'Model: llama-3.2-1b-instruct'),
    tabs: modelTabs,
  },
};

export const WithActions: Story = {
  args: {
    ...Default.args,
    slotActions: <Button color="brand">Open Job</Button>,
  },
};

export const DisabledTabs: Story = {
  args: {
    heading: 'llama-3.2-1b-support',
    slotStatus: <DeploymentStatusBadge state={{ kind: 'loading' }} isAdapter={false} />,
    tabs: modelTabs.map((tab) =>
      tab.value === 'training' || tab.value === 'logs' ? { ...tab, disabled: true } : tab
    ),
  },
};

export const Minimal: Story = {
  args: {
    heading: 'llama-3.2-1b-instruct',
    tabs: modelTabs.slice(0, 2),
  },
};

export const LongHeading: Story = {
  args: {
    ...Default.args,
    heading:
      'llama-3.2-1b-instruct-support-tickets-billing-and-refunds-sft-run-2026-10-19-final-v3',
  },
  decorators: [
    (Story) => (
      <div className="max-w-[640px]">
        <Story />
      </div>
    ),
  ],
};
