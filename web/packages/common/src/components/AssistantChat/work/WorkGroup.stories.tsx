// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ThoughtStep } from '@nemo/common/src/components/AssistantChat/work/ThoughtStep';
import { ToolCallStep } from '@nemo/common/src/components/AssistantChat/work/ToolCallStep';
import { WorkGroupView } from '@nemo/common/src/components/AssistantChat/work/WorkGroup';
import { TooltipProvider } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';

const meta = {
  component: WorkGroupView,
  title: 'UI/AssistantChat/WorkGroup',
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-[560px]">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  args: {
    hasToolCalls: true,
    label: 'Ran 2 tool calls',
    children: (
      <>
        <ThoughtStep text="The user wants tomorrow's weather, so I need the forecast for Paris." />
        <ToolCallStep
          toolName="get_weather"
          args={{ city: 'Paris', date: '2026-11-05' }}
          result={{ forecast: 'Light rain', high_c: 12, precipitation_chance: 0.7 }}
        />
        <ToolCallStep
          toolName="Write"
          args={{ file_path: 'MEMORY.md', content: '- User is in Paris\n' }}
          result="ok"
        />
      </>
    ),
  },
} satisfies Meta<typeof WorkGroupView>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Thinking: Story = {
  args: {
    hasToolCalls: false,
    label: 'Thinking…',
    children: <ToolCallStep toolName="WebSearch" args={{ query: 'Paris weather' }} running />,
  },
};
