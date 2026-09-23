// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Flex, Spinner } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { Box, Plus, Sparkles, Upload } from 'lucide-react';

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
    tag: { label: 'Advanced', color: 'gray', kind: 'solid' } as const,
    enabled: true,
  },
  {
    id: 'upload',
    title: 'Start from a config you already have',
    description: 'Upload a YAML or JSON config and continue from it.',
    icon: Upload,
    tag: { label: 'Intermediate', color: 'gray', kind: 'solid' } as const,
    enabled: true,
  },
  {
    id: 'ai',
    title: 'Describe it and let AI draft it',
    description: 'Say what you want in a sentence and start from the draft.',
    icon: Sparkles,
    tag: { label: 'Beginner', color: 'gray', kind: 'solid' } as const,
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
        templatesTag={{ label: 'Intermediate', color: 'gray', kind: 'solid' }}
        onSelect={() => undefined}
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

/** A group still fetching holds its place rather than appearing later and shifting the page. */
export const GroupStillLoading: Story = {
  render: () => (
    <Demo
      templateGroups={[
        group('a', 'Loaded Group', ['Template Name', 'Template Name']),
        { id: 'b', title: 'Still Loading', templates: [], loading: true },
      ]}
    />
  ),
};

/** Per-group icon colour. What it signifies is the caller's to decide. */
export const AccentedGroups: Story = {
  render: () => (
    <Demo
      templateGroups={[
        { ...group('a', 'Purple', ['Template Name', 'Template Name']), accent: '#b78bf7' },
        { ...group('b', 'Orange', ['Template Name']), accent: '#f0883e' },
        { ...group('c', 'Teal', ['Template Name', 'Template Name']), accent: '#4dd4c1' },
      ]}
    />
  ),
};

/** Locked while the picked entry point is being acted on, with progress in the banner. */
export const Working: Story = {
  render: () => (
    <Demo
      disabled
      slotBanner={
        <Banner kind="inline" status="info">
          <Flex gap="density-md" align="center">
            <Spinner size="small" aria-label="Setting up" />
            Setting up…
          </Flex>
        </Banner>
      }
    />
  ),
};
