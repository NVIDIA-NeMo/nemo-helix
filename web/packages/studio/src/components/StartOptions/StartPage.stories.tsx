// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from '@storybook/react';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import { Box, LayoutTemplate, Plus, Sparkles } from 'lucide-react';
import { useState } from 'react';

const meta: Meta<typeof StartPage> = {
  component: StartPage,
  title: 'Studio/StartPage',
  parameters: { layout: 'fullscreen' },
};

export default meta;

const OPTIONS = [
  {
    id: 'ai',
    title: 'Describe with AI',
    description: 'Say what you want in a sentence and start from the draft.',
    icon: Sparkles,
    tag: { label: 'Beginner', color: 'gray', kind: 'solid' } as const,
    enabled: true,
  },
  {
    id: 'template',
    title: 'Start from a template',
    description: 'Begin from a ready-made recipe and adjust it.',
    icon: LayoutTemplate,
    tag: { label: 'Intermediate', color: 'gray', kind: 'solid' } as const,
    enabled: true,
  },
  {
    id: 'scratch',
    title: 'Build from scratch',
    description: 'Open the full form and choose everything yourself.',
    icon: Plus,
    tag: { label: 'Advanced', color: 'gray', kind: 'solid' } as const,
    enabled: true,
  },
];

const group = (id: string, title: string, names: string[]) => ({
  id,
  title,
  templates: names.map((name) => ({
    id: `${id}-${name}`,
    name,
    description: 'What this template sets up, in a line.',
    icon: Box,
  })),
});

const GROUPS = [
  group('a', 'Template Group Title', ['Template Name', 'Template Name', 'Template Name']),
  group('b', 'Template Group Title', ['Template Name', 'Template Name']),
];

const Demo = (args: Partial<React.ComponentProps<typeof StartPage>>) => {
  const [value, setValue] = useState<string>('template');
  const [templateId, setTemplateId] = useState<string | null>(null);
  return (
    <div className="h-screen">
      <StartPage
        heading="Page Title"
        headingDescription="Page Description"
        options={OPTIONS}
        value={value}
        onChange={(next) => {
          setValue(next);
          setTemplateId(null);
        }}
        canContinue={value !== 'template' || templateId !== null}
        onContinue={() => undefined}
        blockedHint={value === 'template' ? 'Pick a recipe to continue.' : undefined}
        slotDetail={
          value === 'template' ? (
            <TemplateGroups groups={GROUPS} value={templateId} onChange={setTemplateId} />
          ) : null
        }
        {...args}
      />
    </div>
  );
};

type Story = StoryObj<typeof StartPage>;

/** Opens on the template rung, with its recipes revealed below. */
export const Default: Story = { render: () => <Demo /> };

/** A different rung selected: the recipes give way to that option's own panel. */
export const OtherOptionSelected: Story = {
  render: () => <Demo slotDetail={null} value="scratch" canContinue blockedHint={undefined} />,
};

/** Locked while the picked entry point is being acted on. */
export const Working: Story = {
  render: () => <Demo disabled continueLoading canContinue={false} continueLabel="Setting up…" />,
};
