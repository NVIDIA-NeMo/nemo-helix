// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import { MockToastProvider } from '@nemo/common/src/tests/MockToastProvider';
import { AssistantToolCallPart } from '@studio/routes/agents/AssistantChatRoute/AssistantToolCallPart';
import {
  ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME,
  ASSISTANT_JOB_PROGRESS_TOOL_NAME,
} from '@studio/routes/agents/AssistantChatRoute/jobProgressConsts';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';

vi.mock('@studio/routes/agents/AssistantChatRoute/JobProgressToolCall', () => ({
  JobProgressToolCall: ({ args }: { args: Record<string, unknown> }) => (
    <div data-testid="mock-job-progress">{String(args.job_name)}</div>
  ),
}));

type PartProps = ComponentProps<typeof AssistantToolCallPart>;

const renderPart = async ({
  args,
  toolName,
  ...overrides
}: Pick<PartProps, 'args' | 'toolName'> & Partial<PartProps>) => {
  render(
    <MockToastProvider>
      <AssistantToolCallPart
        addResult={vi.fn()}
        args={args}
        argsText={JSON.stringify(args)}
        resume={vi.fn()}
        status={{ type: 'complete' }}
        toolCallId={`toolu_${toolName}`}
        toolName={toolName}
        type="tool-call"
        {...overrides}
      />
    </MockToastProvider>
  );
  await waitFor(() => {
    for (const code of screen.queryAllByTestId('nv-code-snippet-code')) {
      expect(code).not.toHaveClass('nv-code-snippet-code--loading');
    }
  });
};

const getStepTitles = (container: HTMLElement = document.body) =>
  within(container)
    .getAllByTestId('assistant-work-tool-call')
    .map((step) => within(step).getByTestId('nv-collapsible-trigger').textContent);

describe('AssistantToolCallPart', () => {
  it.each([ASSISTANT_THINKING_TOOL_NAME, ASSISTANT_COLLAPSED_THINKING_TOOL_NAME])(
    'renders %s as a Thought step',
    async (toolName) => {
      await renderPart({ args: { text: 'Working out which file owns the panel.' }, toolName });

      expect(screen.getByTestId('assistant-work-thought')).toHaveTextContent(
        'ThoughtWorking out which file owns the panel.'
      );
    }
  );

  it('renders an unknown tool as a Used step with its arguments and result', async () => {
    await renderPart({
      args: { query: 'symbols' },
      result: { matches: 3 },
      toolName: 'InspectWorkspace',
    });

    const step = screen.getByTestId('assistant-work-tool-call');
    expect(getStepTitles()).toEqual(['Used InspectWorkspace symbols']);
    expect(within(step).getByText('Arguments')).toBeInTheDocument();
    expect(within(step).getByText('Result')).toBeInTheDocument();
  });

  it('unpacks a subtle tool group into one step per call', async () => {
    await renderPart({
      args: {
        actions: [
          { args: { command: 'pwd' }, toolCallId: 'toolu_bash', toolName: 'Bash' },
          {
            args: { file_path: 'web/packages/studio/src/App.tsx' },
            toolCallId: 'toolu_read',
            toolName: 'Read',
          },
          { args: { pattern: 'TODO' }, toolCallId: 'toolu_grep', toolName: 'Grep' },
        ],
      },
      toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
    });

    expect(getStepTitles()).toEqual(['Ran pwd', 'Read App.tsx', 'Searched text TODO']);
    expect(screen.queryByTestId('assistant-work-repeated-tool-calls')).not.toBeInTheDocument();
  });

  it('shows the last call of a running subtle group with its running action', async () => {
    await renderPart({
      args: {
        actions: [
          { args: { file_path: 'a.md' }, toolCallId: 'toolu_read', toolName: 'Read' },
          { args: { command: 'pwd' }, toolCallId: 'toolu_bash', toolName: 'Bash' },
        ],
      },
      status: { type: 'running' },
      toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
    });

    expect(getStepTitles()).toEqual(['Read a.md', 'Running pwd']);
  });

  it('folds repeated calls of one tool into a step per tool, keeping each command', async () => {
    const user = userEvent.setup();
    await renderPart({
      args: {
        actions: [
          {
            args: { command: 'pwd', description: 'check working directory' },
            toolCallId: 'toolu_bash_1',
            toolName: 'Bash',
          },
          {
            args: { command: 'ls', description: 'list files' },
            toolCallId: 'toolu_bash_2',
            toolName: 'Bash',
          },
          { args: { file_path: 'README.md' }, toolCallId: 'toolu_read_1', toolName: 'Read' },
          {
            args: { command: 'git status', description: 'check git status' },
            toolCallId: 'toolu_bash_3',
            toolName: 'Bash',
          },
          {
            args: { command: 'pnpm test', description: 'run tests' },
            toolCallId: 'toolu_bash_4',
            toolName: 'Bash',
          },
          { args: { command: 'pnpm typecheck' }, toolCallId: 'toolu_bash_5', toolName: 'Bash' },
          { args: { file_path: 'package.json' }, toolCallId: 'toolu_read_2', toolName: 'Read' },
          { args: { pattern: 'x', path: 'src' }, toolCallId: 'toolu_grep', toolName: 'Grep' },
        ],
      },
      toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
    });

    const [commands, reads] = screen.getAllByTestId('assistant-work-repeated-tool-calls');
    expect(commands).toHaveTextContent('Ran 5 commands');
    expect(reads).toHaveTextContent('Read 2 files');
    expect(commands).not.toHaveAttribute('open');
    expect(getStepTitles(commands)).toEqual([
      'Ran check working directory',
      'Ran list files',
      'Ran check git status',
      'Ran run tests',
      'Ran pnpm typecheck',
    ]);
    expect(getStepTitles(reads)).toEqual(['Read README.md', 'Read package.json']);
    expect(getStepTitles()).toContain('Searched text x in src');

    await user.click(within(commands).getByText('Ran 5 commands'));

    expect(commands).toHaveAttribute('open');
    const [, listFiles] = within(commands).getAllByTestId('assistant-work-tool-call');
    await user.click(within(listFiles!).getByText('list files'));

    expect(within(listFiles!).getByText('Command')).toBeVisible();
    expect(listFiles).toHaveTextContent('ls');
  });

  it('unpacks collapsed details into thought, markdown narration, and tool steps', async () => {
    await renderPart({
      args: {
        label: 'worked for 42s',
        parts: [
          { type: 'reasoning', text: 'Plan the optimization report.' },
          {
            type: 'text',
            text: [
              '## Optimization report',
              '',
              '**Current config:** `meta-llama-3-1-70b-instruct`',
              '',
              '- Preserved first suggestion',
              '- Preserved second suggestion',
            ].join('\n'),
          },
          {
            type: 'tool-call',
            args: { command: 'pwd' },
            argsText: '{"command":"pwd"}',
            toolCallId: 'toolu_bash',
            toolName: 'Bash',
          },
          {
            type: 'tool-call',
            args: {
              actions: [
                { args: { file_path: 'a.md' }, toolCallId: 'toolu_read_1', toolName: 'Read' },
                { args: { file_path: 'b.md' }, toolCallId: 'toolu_read_2', toolName: 'Read' },
              ],
            },
            toolCallId: 'toolu_group',
            toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
          },
        ],
      },
      argsText: '',
      toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
    });

    expect(screen.getByTestId('assistant-work-thought')).toHaveTextContent(
      'Plan the optimization report.'
    );
    expect(screen.getByRole('heading', { level: 2, name: 'Optimization report' })).toBeVisible();
    expect(screen.getByText('Current config:')).toHaveProperty('tagName', 'STRONG');
    expect(screen.getByText('meta-llama-3-1-70b-instruct')).toHaveProperty('tagName', 'CODE');
    expect(screen.getAllByRole('listitem')[0]).toHaveTextContent('Preserved first suggestion');
    expect(getStepTitles()).toEqual(['Ran pwd', 'Read a.md', 'Read b.md']);
    expect(screen.getByTestId('assistant-work-repeated-tool-calls')).toHaveTextContent(
      'Read 2 files'
    );
    expect(screen.queryByText(/worked for/)).not.toBeInTheDocument();
  });

  it.each([ASSISTANT_JOB_PROGRESS_TOOL_NAME, ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME])(
    'renders %s as a rich job progress tool call',
    async (toolName) => {
      await renderPart({ args: { job_name: 'studio-job-1' }, toolName });

      expect(screen.getByTestId('mock-job-progress')).toHaveTextContent('studio-job-1');
      expect(screen.queryByTestId('assistant-work-tool-call')).not.toBeInTheDocument();
    }
  );
});
