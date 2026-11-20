// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { MessageStatus, ThreadMessageLike } from '@assistant-ui/react';
import { AssistantChat } from '@nemo/common/src/components/AssistantChat';
import {
  CONTENT_FILTERED_STATUS,
  EMPTY_REPLY_STATUS,
  ENDED_EARLY_STATUS,
  TRUNCATED_STATUS,
} from '@nemo/common/src/components/AssistantChat/constants';
import { TooltipProvider } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';

// Review the notice shown when a model stops before finishing its reply. Each scenario maps to the
// stream's finish_reason (or its absence); the wording lives in `constants.ts`.

const PROMPT = 'Describe a Christmas 90s cartoon animation logo for NeMo Studio.';
const REASONING =
  'The user wants a logo description. Think about 90s cartoon openers: thick outlines, saturated colors, bouncy lettering, snow, candy-cane stripes...';
const PARTIAL = 'Picture chunky, bouncing letters spelling "NeMo Studio" in candy-cane red and';

type Part = Exclude<ThreadMessageLike['content'], string>[number];

const SCENARIOS: Record<string, { label: string; parts: Part[]; status: MessageStatus }> = {
  truncated: {
    label: 'length: token limit hit while reasoning',
    parts: [{ type: 'reasoning', text: REASONING }],
    status: TRUNCATED_STATUS,
  },
  'content-filter': {
    label: 'content_filter: stopped by provider filter',
    parts: [{ type: 'text', text: PARTIAL }],
    status: CONTENT_FILTERED_STATUS,
  },
  'ended-early': {
    label: 'no finish_reason: stream cut off',
    parts: [{ type: 'text', text: PARTIAL }],
    status: ENDED_EARLY_STATUS,
  },
  empty: {
    label: 'stop: reasoning but no answer',
    parts: [{ type: 'reasoning', text: REASONING }],
    status: EMPTY_REPLY_STATUS,
  },
};

type ScenarioKey = keyof typeof SCENARIOS;

const meta = {
  title: 'UI/AssistantChat/Stopped early',
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="h-[520px] max-w-[760px]">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  args: { scenario: 'truncated' },
  argTypes: {
    scenario: {
      control: {
        type: 'select',
        labels: Object.fromEntries(
          Object.entries(SCENARIOS).map(([key, { label }]) => [key, label])
        ),
      },
      options: Object.keys(SCENARIOS),
    },
  },
  render: ({ scenario }) => {
    const { parts, status } = SCENARIOS[scenario];
    const messages: readonly ThreadMessageLike[] = [
      { role: 'user', content: [{ type: 'text', text: PROMPT }] },
      { role: 'assistant', content: parts, status },
    ];
    // initialMessages only seeds the thread, so remount when the scenario changes.
    return <AssistantChat key={scenario} model="story-model" initialMessages={messages} disabled />;
  },
} satisfies Meta<{ scenario: ScenarioKey }>;

export default meta;
type Story = StoryObj<typeof meta>;

export const StoppedEarly: Story = {};
