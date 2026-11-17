// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RepeatedToolCallsStep } from '@nemo/common/src/components/AssistantChat/work/RepeatedToolCallsStep';
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
    label: 'Ran 2 tool calls · worked for 42s',
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

export const Collapsed: Story = {};

export const Expanded: Story = {
  args: {
    defaultOpen: true,
    children: (
      <>
        <ThoughtStep
          defaultOpen
          text="The user wants tomorrow's weather, so I need the forecast for Paris."
        />
        <ToolCallStep
          defaultOpen
          toolName="get_weather"
          args={{ city: 'Paris', date: '2026-11-05' }}
          result={{ forecast: 'Light rain', high_c: 12, precipitation_chance: 0.7 }}
        />
        <ToolCallStep
          toolName="Bash"
          args={{ command: 'cat MEMORY.md', description: 'read saved notes' }}
          isError
          result="cat: MEMORY.md: No such file or directory"
        />
      </>
    ),
  },
};

export const Running: Story = {
  args: {
    hasToolCalls: true,
    label: 'Thinking…',
    defaultOpen: true,
    children: (
      <>
        <ToolCallStep toolName="Read" args={{ file_path: 'docs/forecast.md' }} />
        <ToolCallStep toolName="WebSearch" args={{ query: 'Paris weather' }} running />
      </>
    ),
  },
};

export const RepeatedCalls: Story = {
  args: {
    defaultOpen: true,
    label: 'Ran 3 tool calls',
    children: (
      <RepeatedToolCallsStep defaultOpen title="Read 3 files" toolName="Read">
        <ToolCallStep toolName="Read" args={{ file_path: 'README.md' }} />
        <ToolCallStep toolName="Read" args={{ file_path: 'package.json' }} />
        <ToolCallStep toolName="Read" args={{ file_path: 'src/app.tsx' }} />
      </RepeatedToolCallsStep>
    ),
  },
};

export const ThoughtOnly: Story = {
  args: {
    hasToolCalls: false,
    label: 'Thought',
    children: <ThoughtStep text="A short greeting needs no tools." />,
  },
};
