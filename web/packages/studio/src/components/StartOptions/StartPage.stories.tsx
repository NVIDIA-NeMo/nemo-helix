// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from '@storybook/react';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { StartSubPage } from '@studio/components/StartOptions/StartSubPage';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import { Box, LayoutTemplate, Plus, Sparkles } from 'lucide-react';

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
  group('a', 'Template Group Title', ['Template One', 'Template Two', 'Template Three']),
  group('b', 'Template Group Title', ['Template Four', 'Template Five']),
];

const Demo = ({
  pendingId = null,
  ...args
}: Partial<React.ComponentProps<typeof StartPage>> & { pendingId?: string | null }) => (
  <div className="h-screen">
    <StartPage
      heading="Page Title"
      headingDescription="Page Description"
      options={OPTIONS}
      value="template"
      onChange={() => undefined}
      slotDetail={
        <TemplateGroups
          groups={GROUPS}
          onSelect={() => undefined}
          pendingId={pendingId}
          pendingLabel="Registering model…"
          disabled={pendingId !== null}
        />
      }
      disabled={pendingId !== null}
      {...args}
    />
  </div>
);

type Story = StoryObj<typeof StartPage>;

/** Opens on the template rung, with its recipes below. Every tile acts on click. */
export const Default: Story = { render: () => <Demo /> };

/** A recipe being set up: its tile says what is running, and every tile waits. */
export const SettingUp: Story = { render: () => <Demo pendingId="a-Template One" /> };

/** An option's own step, reached from its tile, with Back and a gated Continue. */
export const SubPage: StoryObj<typeof StartSubPage> = {
  render: () => (
    <div className="h-screen">
      <StartSubPage
        heading="Describe with AI"
        headingDescription="Say what you want in a sentence and start from the draft."
        onBack={() => undefined}
        canContinue={false}
        onContinue={() => undefined}
        blockedHint="Generate a valid config to continue."
      >
        <div>Option content</div>
      </StartSubPage>
    </div>
  ),
};
