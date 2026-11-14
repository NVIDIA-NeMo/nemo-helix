// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Button, TabsContent, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { DetailPageLayout, type DetailPageTab } from '@studio/components/Layouts/DetailPageLayout';
import { useState } from 'react';

type Tab = 'details' | 'chat' | 'evaluations' | 'logs';

const TABS: DetailPageTab<Tab>[] = [
  { value: 'details', label: 'Details' },
  { value: 'chat', label: 'Chat' },
  { value: 'evaluations', label: 'Evaluations' },
  { value: 'logs', label: 'Logs', disabled: true },
];

interface StoryArgs {
  title: string;
  description?: string;
  showBadges: boolean;
  showActions: boolean;
}

const DetailPageLayoutStory = ({ title, description, showBadges, showActions }: StoryArgs) => {
  const [selectedTab, setSelectedTab] = useState<Tab>('details');
  return (
    <div className="h-[480px]">
      <DetailPageLayout
        title={title}
        slotDescription={description}
        slotBadges={
          showBadges && (
            <>
              <Badge kind="solid" color="green">
                Deployed
              </Badge>
              <Badge kind="solid" color="gray">
                LoRA
              </Badge>
            </>
          )
        }
        slotActions={showActions && <Button color="brand">Run Evaluation</Button>}
        tabs={TABS}
        selectedTab={selectedTab}
        onTabChange={setSelectedTab}
      >
        {TABS.map((tab) => (
          <TabsContent key={tab.value} value={tab.value} className="p-0 pt-6">
            <Text>{tab.label} content</Text>
          </TabsContent>
        ))}
      </DetailPageLayout>
    </div>
  );
};

const meta = {
  component: DetailPageLayoutStory,
  title: 'Studio/Layouts/DetailPageLayout',
  args: {
    title: 'llama-3.1-8b-instruct-lora',
    showBadges: true,
    showActions: true,
  },
} satisfies Meta<typeof DetailPageLayoutStory>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithDescription: Story = {
  args: { description: 'Fine-tuned on the customer support dataset.' },
};

export const LongTitle: Story = {
  args: {
    title:
      'an-extremely-long-entity-name-that-does-not-fit-on-one-line-alongside-its-status-badges-and-tags',
  },
};

export const Minimal: Story = {
  args: { showBadges: false, showActions: false },
};
