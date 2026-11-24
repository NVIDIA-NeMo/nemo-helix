// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getSubtleToolGroupActions,
  groupRepeatedSubtleToolActions,
} from '@nemo/common/src/components/AssistantChat/toolCall/helpers';
import { RepeatedToolCallsStep } from '@nemo/common/src/components/AssistantChat/work/RepeatedToolCallsStep';
import { ToolCallStep } from '@nemo/common/src/components/AssistantChat/work/ToolCallStep';
import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import { Stack, Text, TooltipProvider } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import type { ComponentProps } from 'react';

type ToolCallStepProps = ComponentProps<typeof ToolCallStep>;

const meta = {
  component: ToolCallStep,
  title: 'UI/AssistantChat/ToolCallStep',
  decorators: [
    (Story) => (
      <ToastProvider>
        <TooltipProvider>
          <div className="max-w-[560px]">
            <Story />
          </div>
        </TooltipProvider>
      </ToastProvider>
    ),
  ],
  args: {
    toolName: 'mcp__browser__web_search',
    args: { query: 'kaizen ui collapsible' },
    result: { results: 3 },
  },
} satisfies Meta<typeof ToolCallStep>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Labels each step with the tool name it was called with, so the matched step reads against it. */
const LabeledSteps = ({ steps }: { readonly steps: readonly ToolCallStepProps[] }) => (
  <Stack gap="density-lg">
    {steps.map((step) => (
      <Stack key={step.toolName} gap="density-xs">
        <Text kind="mono/sm" className="text-subtle">
          {step.toolName}
        </Text>
        <ToolCallStep {...step} />
      </Stack>
    ))}
  </Stack>
);

const NAMING_VARIANTS: readonly ToolCallStepProps[] = [
  { toolName: 'WebSearch', args: { query: 'kaizen ui' } },
  { toolName: 'web_search', args: { query: 'kaizen ui' } },
  { toolName: 'search-web', args: { query: 'kaizen ui' } },
  { toolName: 'mcp__browser__web_search', args: { query: 'kaizen ui' } },
  { toolName: 'read_file', args: { path: 'web/packages/studio/src/App.tsx' } },
  { toolName: 'mcp__filesystem__read_file', args: { path: 'README.md' } },
  { toolName: 'list_directory', args: { path: 'web/packages' } },
  { toolName: 'run_command', args: { command: 'ls -la' } },
  { toolName: 'fetch_url', args: { url: 'https://docs.nvidia.com' } },
];

/** Edit any field in Controls, such as `toolName`, to see which step a name matches. */
export const Playground: Story = {};

/** Naming styles, MCP prefixes, and aliases that match a built-in tool's step. */
export const NamingVariants: Story = {
  render: () => <LabeledSteps steps={NAMING_VARIANTS} />,
};

export const RunningNamingVariants: Story = {
  render: () => (
    <LabeledSteps steps={NAMING_VARIANTS.map((step) => ({ ...step, running: true }))} />
  ),
};

/** Aliased write and edit tools render as file changes; a failed write keeps its error. */
export const AliasedFileChanges: Story = {
  render: () => (
    <LabeledSteps
      steps={[
        {
          toolName: 'mcp__filesystem__write_file',
          args: { path: 'notes/todo.md', content: '- Review tool steps\n- Ship it\n' },
        },
        {
          toolName: 'edit_file',
          args: {
            file_path: 'src/config.ts',
            old_string: 'const retries = 1;\n',
            new_string: 'const retries = 3;\nconst backoffMs = 500;\n',
          },
        },
        {
          toolName: 'write_file',
          args: { path: 'notes/todo.md', content: 'Pick up milk' },
          isError: true,
          result: 'EACCES: permission denied',
        },
      ]}
    />
  ),
};

/** Variants of one tool fold into a single repeated step. */
export const GroupedVariants: Story = {
  render: () => {
    const actions = getSubtleToolGroupActions({
      actions: [
        { toolCallId: 'a', toolName: 'read_file', args: { path: 'src/a.ts' } },
        { toolCallId: 'b', toolName: 'mcp__fs__Read', args: { path: 'src/b.ts' } },
        { toolCallId: 'c', toolName: 'Read', args: { file_path: 'src/c.ts' } },
      ],
    });
    return (
      <>
        {groupRepeatedSubtleToolActions(actions).map((group) => (
          <RepeatedToolCallsStep
            key={group.actions[0].toolCallId}
            title={group.repeatedTitle ?? ''}
            toolName={group.actions[0].toolName}
          >
            {group.actions.map((action) => (
              <ToolCallStep key={action.toolCallId} args={action.args} toolName={action.toolName} />
            ))}
          </RepeatedToolCallsStep>
        ))}
      </>
    );
  },
};

/** Names that match no built-in tool keep the generic Used step. */
export const UnmatchedNames: Story = {
  render: () => (
    <LabeledSteps
      steps={[
        { toolName: 'get_weather', args: { city: 'Paris' }, result: { forecast: 'Light rain' } },
        { toolName: 'save_note', args: { file_path: 'MEMORY.md', content: 'Pick up milk' } },
        { toolName: 'read_files_batch', args: { paths: ['a.ts', 'b.ts'] } },
      ]}
    />
  ),
};
