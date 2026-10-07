// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { Stack, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryContext, StoryObj } from '@storybook/react';
import {
  QuickstartSamplePanel,
  type QuickstartSampleAgent,
} from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel';
import type { ComponentType } from 'react';

const SAMPLE_AGENT: QuickstartSampleAgent = {
  name: 'email-security-triage',
  description: 'Triages inbound email for phishing, spoofing, and social-engineering signals.',
  status: 'Running',
  // Deliberately not the agent name — deployments are `${agent}-${8 hex}`.
  deploymentName: 'email-security-triage-9f2a1c',
};

/**
 * Keyed on `defaultView`, which only seeds internal state, so without a remount the
 * control would be a no-op. `max-w` because a hard 1152px overflows the docs iframe.
 */
const panelDecorator = [
  (Story: ComponentType, context: StoryContext) => (
    <div key={String(context.args.defaultView)} className="w-full max-w-[1152px]">
      <Story />
    </div>
  ),
];

const meta = {
  component: QuickstartSamplePanel,
  title: 'Routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel',
  decorators: panelDecorator,
  args: {
    workspace: DEFAULT_WORKSPACE,
    agent: SAMPLE_AGENT,
    defaultView: 'studio',
    onSwitchWorkspace: () => {},
  },
  argTypes: {
    defaultView: {
      name: 'View',
      control: 'inline-radio',
      options: ['studio', 'cli'],
      description: 'Initial tab. The panel is uncontrolled, so this only seeds the first render.',
    },
  },
} satisfies Meta<typeof QuickstartSamplePanel>;

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Studio view. Steps 1–2 carry a single action while steps 3–4 carry two, so the
 * one-action and two-action layouts are both visible here without a separate story.
 */
export const Studio: Story = {};

/**
 * NeMo CLI view — action buttons give way to copyable `nemo` commands. Static rather than
 * click-driven, so the a11y addon (which audits at mount) actually scans it.
 */
export const Cli: Story = {
  args: { defaultView: 'cli' },
};

/**
 * No sample agent is installed, so the section is hidden entirely. Rendered between
 * two markers — an empty canvas is otherwise indistinguishable from a broken story.
 */
export const NoSampleAgent: Story = {
  args: { agent: undefined },
  render: (args) => (
    <Stack gap="density-lg" className="w-full">
      <Text kind="body/regular/sm" className="text-secondary">
        Dashboard content above the Quickstart panel.
      </Text>
      <QuickstartSamplePanel {...args} />
      <Text kind="body/regular/sm" className="text-secondary">
        Dashboard content below — these two lines should sit flush, with nothing between.
      </Text>
    </Stack>
  ),
};

/**
 * Intake and agent optimizations turned off, so steps 2 and 4 are dropped rather than left
 * pointing at routes this deployment never registers. The timeline must still close cleanly
 * on whatever step ends up last.
 */
export const FeaturesDisabled: Story = {
  args: { intakeEnabled: false, agentOptimizationsEnabled: false },
};

/**
 * happy-dom has no layout engine, so single-line truncation can only be proven
 * visually. The badge and chevron must stay pinned right however long the text runs.
 */
export const LongAgentDescription: Story = {
  args: {
    agent: {
      ...SAMPLE_AGENT,
      name: 'email-security-triage-with-an-unusually-long-generated-suffix-9f2a1c',
      description:
        'Triages inbound email for phishing, spoofing, social-engineering, credential-harvesting, and business-email-compromise signals, then routes each verdict to the appropriate downstream queue with a confidence score and a short rationale.',
    },
  },
};

/** Guards against the status badge being hardcoded to a green "Running". */
export const AgentNotRunning: Story = {
  args: {
    agent: { ...SAMPLE_AGENT, status: 'Failed' },
  },
};

/**
 * Phone width. Each step's actions drop to their own row so the title and description get
 * the full width; the footer wraps and the CLI snippets break rather than scroll.
 */
export const NarrowContainer: Story = {
  decorators: [
    (Story: ComponentType) => (
      <div className="w-[360px]">
        <Story />
      </div>
    ),
  ],
};
