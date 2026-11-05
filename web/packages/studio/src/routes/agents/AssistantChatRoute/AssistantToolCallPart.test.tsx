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

  it('labels a running tool call as Calling and a failed result as Error', () => {
    renderPart({
      args: { command: 'pwd' },
      isError: true,
      result: 'permission denied',
      status: { type: 'running' },
      toolName: 'Bash',
    });

    const step = screen.getByTestId('assistant-work-tool-call');
    expect(step).toHaveTextContent('Calling Bash');
    expect(within(step).getByText('Error')).toHaveClass('text-feedback-danger');
  });

  it('renders a tool with a file path and content as an Edited step', () => {
    renderPart({
      args: { file_path: 'src/notes/MEMORY.md', content: 'Pick up milk' },
      toolName: 'Write',
    });

    const step = screen.getByTestId('assistant-work-file-edit');
    expect(step).toHaveTextContent('Edited MEMORY.md');
    expect(within(step).getByText('src/notes/MEMORY.md')).toBeInTheDocument();
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
        expect.stringContaining('Called Read'),
        expect.stringContaining('Called Grep'),
      ])
    );
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
