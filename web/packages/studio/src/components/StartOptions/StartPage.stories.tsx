// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from '@storybook/react';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { Box, Plus, Sparkles, Upload } from 'lucide-react';
import { useState } from 'react';

const meta: Meta<typeof StartPage> = {
  component: StartPage,
  title: 'Studio/StartPage',
  parameters: { layout: 'fullscreen' },
};

export default meta;

const OPTIONS = [
  {
    id: 'scratch',
    title: 'Build from scratch',
    description: 'Open the full form with sensible defaults and choose everything yourself.',
    icon: Plus,
    enabled: true,
  },
  {
    id: 'upload',
    title: 'Start from a config you already have',
    description: 'Upload a YAML or JSON config and continue from it.',
    icon: Upload,
    enabled: true,
  },
  {
    id: 'ai',
    title: 'Describe it and let AI draft it',
    description: 'Say what you want in a sentence and start from the draft.',
    icon: Sparkles,
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

const Demo = (args: Partial<React.ComponentProps<typeof StartPage>>) => {
  const [value, setValue] = useState<string | null>(null);
  return (
    <div className="h-screen">
      <StartPage
        heading="Page Title"
        headingDescription="Page Description"
        options={OPTIONS}
        templateGroups={[
          group('a', 'Template Group Title', ['Template Name', 'Template Name', 'Template Name']),
          group('b', 'Template Group Title', ['Template Name', 'Template Name']),
        ]}
        value={value}
        onChange={setValue}
        canContinue={value !== null}
        onContinue={() => undefined}
        {...args}
      />
    </div>
  );
};

type Story = StoryObj<typeof StartPage>;

/** The full pattern: options, a divider, then grouped templates. */
export const Default: Story = { render: () => <Demo /> };

/** Options only — a flow with nothing to template from yet. */
export const WithoutTemplates: Story = { render: () => <Demo templateGroups={[]} /> };

/** Locked while the picked entry point is being acted on. */
export const Working: Story = {
  render: () => <Demo disabled continueLoading canContinue={false} continueLabel="Setting up…" />,
};
