// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import { AssistantToolCallPart } from '@studio/routes/agents/AssistantChatRoute/AssistantToolCallPart';
import {
  ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME,
  ASSISTANT_JOB_PROGRESS_TOOL_NAME,
} from '@studio/routes/agents/AssistantChatRoute/jobProgressConsts';
import { render, screen, within } from '@testing-library/react';
import type { ComponentProps } from 'react';

vi.mock('@studio/routes/agents/AssistantChatRoute/JobProgressToolCall', () => ({
  JobProgressToolCall: ({ args }: { args: Record<string, unknown> }) => (
    <div data-testid="mock-job-progress">{String(args.job_name)}</div>
  ),
}));

type PartProps = ComponentProps<typeof AssistantToolCallPart>;

const renderPart = ({
  args,
  toolName,
  ...overrides
}: Pick<PartProps, 'args' | 'toolName'> & Partial<PartProps>) =>
  render(
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
  );

describe('AssistantToolCallPart', () => {
  it.each([ASSISTANT_THINKING_TOOL_NAME, ASSISTANT_COLLAPSED_THINKING_TOOL_NAME])(
    'renders %s as a Thought step',
    (toolName) => {
      renderPart({ args: { text: 'Working out which file owns the panel.' }, toolName });

      expect(screen.getByTestId('assistant-work-thought')).toHaveTextContent(
        'ThoughtWorking out which file owns the panel.'
      );
    }
  );

  it('renders a regular tool call as a Called step with its arguments and result', () => {
    renderPart({
      args: { query: 'symbols' },
      result: { matches: 3 },
      toolName: 'InspectWorkspace',
    });

    const step = screen.getByTestId('assistant-work-tool-call');
    expect(step).toHaveTextContent('Called InspectWorkspace');
    expect(within(step).getByText('Arguments')).toBeInTheDocument();
    expect(within(step).getByText('Result')).toBeInTheDocument();
  });

  it('labels a running known tool with its running action and a failed result as Error', () => {
    renderPart({
      args: { command: 'pwd' },
      isError: true,
      result: 'permission denied',
      status: { type: 'running' },
      toolName: 'Bash',
    });

    const step = screen.getByTestId('assistant-work-tool-call');
    expect(step).toHaveTextContent('Running command pwd');
    expect(within(step).getByText('Error')).toHaveClass('text-feedback-danger');
  });

  it('renders a known file tool as a file step with its action text', () => {
    renderPart({
      args: { file_path: 'src/notes/MEMORY.md', content: 'Pick up milk' },
      toolName: 'Write',
    });

    const step = screen.getByTestId('assistant-work-file-edit');
    expect(step).toHaveTextContent('Wrote file MEMORY.md');
    expect(within(step).getByText('src/notes/MEMORY.md')).toBeInTheDocument();
  });

  it('shows line counts and the arguments of an edit', () => {
    renderPart({
      args: { file_path: 'src/app.ts', old_string: 'a\nb', new_string: 'c' },
      toolName: 'Edit',
    });

    const step = screen.getByTestId('assistant-work-file-edit');
    expect(step).toHaveTextContent('Edited file app.ts');
    expect(within(step).getByTestId('assistant-work-edit-stats')).toHaveTextContent('+1 −2');
    expect(within(step).getByText('Arguments')).toBeInTheDocument();
  });

  it('shows a Bash command as a shell snippet rather than JSON', () => {
    renderPart({ args: { command: 'ls -la' }, toolName: 'Bash' });

    const step = screen.getByTestId('assistant-work-tool-call');
    expect(within(step).getByText('Command')).toBeInTheDocument();
    expect(within(step).queryByText('Arguments')).not.toBeInTheDocument();
  });

  it('renders an unknown tool with file-like arguments as a plain Called step', () => {
    renderPart({
      args: { file_path: 'src/notes/MEMORY.md', content: 'Pick up milk' },
      toolName: 'save_note',
    });

    expect(screen.queryByTestId('assistant-work-file-edit')).not.toBeInTheDocument();
    expect(screen.getByTestId('assistant-work-tool-call')).toHaveTextContent('Called save_note');
  });

  it('keeps a failed file write as a tool call so its error shows', () => {
    renderPart({
      args: { file_path: 'src/notes/MEMORY.md', content: 'Pick up milk' },
      isError: true,
      result: 'EACCES: permission denied',
      toolName: 'Write',
    });

    expect(screen.queryByTestId('assistant-work-file-edit')).not.toBeInTheDocument();
    const step = screen.getByTestId('assistant-work-tool-call');
    expect(step).toHaveTextContent('Wrote file MEMORY.md');
    expect(within(step).getByText('Error')).toBeInTheDocument();
  });

  it('unpacks a subtle tool group into one step per call', () => {
    renderPart({
      args: {
        actions: [
          { toolCallId: 'toolu_read', toolName: 'Read', args: { file_path: 'README.md' } },
          { toolCallId: 'toolu_grep', toolName: 'Grep', args: { pattern: 'AssistantChat' } },
        ],
      },
      toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
    });

    const steps = screen.getAllByTestId('assistant-work-tool-call');
    expect(steps.map((step) => step.textContent)).toEqual(
      expect.arrayContaining([
        expect.stringContaining('Read file README.md'),
        expect.stringContaining('Searched text AssistantChat'),
      ])
    );
  });

  it('folds repeated calls of one tool in a subtle group into a single step', () => {
    renderPart({
      args: {
        actions: [
          { toolCallId: 'toolu_read_1', toolName: 'Read', args: { file_path: 'a.md' } },
          { toolCallId: 'toolu_grep', toolName: 'Grep', args: { pattern: 'x', path: 'src' } },
          { toolCallId: 'toolu_read_2', toolName: 'Read', args: { file_path: 'b.md' } },
          { toolCallId: 'toolu_read_3', toolName: 'Read', args: { file_path: 'c.md' } },
        ],
      },
      toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
    });

    const repeated = screen.getByTestId('assistant-work-repeated-tool-calls');
    expect(repeated).toHaveTextContent('Read 3 files');
    expect(within(repeated).getAllByTestId('assistant-work-tool-call')).toHaveLength(3);
    expect(
      screen.getAllByTestId('assistant-work-tool-call').map((step) => step.textContent)
    ).toEqual(expect.arrayContaining([expect.stringContaining('Searched text x in src')]));
  });

  it('unpacks collapsed details into thought, narration, and tool steps', () => {
    renderPart({
      args: {
        label: 'worked for 42s',
        parts: [
          { type: 'reasoning', text: 'Plan the optimization report.' },
          { type: 'text', text: '## Optimization report' },
          {
            type: 'tool-call',
            toolCallId: 'toolu_search',
            toolName: 'search_configs',
            args: { query: 'llama' },
          },
        ],
      },
      toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
    });

    expect(screen.getByTestId('assistant-work-thought')).toHaveTextContent(
      'Plan the optimization report.'
    );
    expect(screen.getByText('Optimization report')).toBeInTheDocument();
    expect(screen.getByTestId('assistant-work-tool-call')).toHaveTextContent(
      'Called search_configs'
    );
  });

  it.each([ASSISTANT_JOB_PROGRESS_TOOL_NAME, ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME])(
    'renders %s as a rich job progress tool call',
    (toolName) => {
      renderPart({ args: { job_name: 'studio-job-1' }, toolName });

      expect(screen.getByTestId('mock-job-progress')).toHaveTextContent('studio-job-1');
      expect(screen.queryByTestId('assistant-work-tool-call')).not.toBeInTheDocument();
    }
  );
});
