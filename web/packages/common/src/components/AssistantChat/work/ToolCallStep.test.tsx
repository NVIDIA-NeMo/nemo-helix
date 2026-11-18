// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ToolCallStep } from '@nemo/common/src/components/AssistantChat/work/ToolCallStep';
import { MockToastProvider } from '@nemo/common/src/tests/MockToastProvider';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';

const renderStep = async (props: ComponentProps<typeof ToolCallStep>) => {
  render(
    <MockToastProvider>
      <ToolCallStep {...props} />
    </MockToastProvider>
  );
  await waitFor(() => {
    for (const code of screen.queryAllByTestId('nv-code-snippet-code')) {
      expect(code).not.toHaveClass('nv-code-snippet-code--loading');
    }
  });
};

const getStepTitle = () =>
  within(screen.getByTestId('assistant-work-tool-call')).getByTestId('nv-collapsible-trigger')
    .textContent;

describe('ToolCallStep', () => {
  it.each([
    {
      args: { command: 'pwd', description: 'check the working directory' },
      expectedText: 'Ran check the working directory',
      toolName: 'Bash',
    },
    {
      args: { question: 'Do you want to continue?' },
      expectedText: 'Asked Do you want to continue?',
      toolName: 'AskUserQuestion',
    },
    {
      args: { query: 'AssistantToolCallPart' },
      expectedText: 'Searched files AssistantToolCallPart',
      toolName: 'FindFiles',
    },
    {
      args: { pattern: 'AssistantToolCallPart' },
      expectedText: 'Searched text AssistantToolCallPart',
      toolName: 'Grep',
    },
    {
      args: { task: 'check the route' },
      expectedText: 'Created task check the route',
      toolName: 'TaskCreate',
    },
    {
      args: { status: 'in_progress' },
      expectedText: 'Updated task in_progress',
      toolName: 'TaskUpdate',
    },
    {
      args: { query: 'read files' },
      expectedText: 'Searched tools read files',
      toolName: 'ToolSearch',
    },
    {
      args: { file_path: 'web/packages/studio/src/App.tsx' },
      expectedText: 'Read App.tsx',
      toolName: 'Read',
    },
    { args: { pattern: '**/*.tsx' }, expectedText: 'Found files **/*.tsx', toolName: 'Glob' },
  ])('renders $toolName as a titled step', async ({ args, expectedText, toolName }) => {
    await renderStep({ args, toolName });

    expect(getStepTitle()).toBe(expectedText);
    expect(screen.queryByTestId('assistant-work-file-edit')).not.toBeInTheDocument();
  });

  it('renders an unknown tool as a Used step with its arguments and result', async () => {
    await renderStep({ args: { query: 'symbols' }, result: { matches: 3 }, toolName: 'Inspect' });

    const step = screen.getByTestId('assistant-work-tool-call');
    expect(getStepTitle()).toBe('Used Inspect symbols');
    expect(within(step).getByText('Arguments')).toBeInTheDocument();
    expect(within(step).getByText('Result')).toBeInTheDocument();
  });

  it('labels a running known tool with its running action and a failed result as Error', async () => {
    await renderStep({
      args: { command: 'pwd' },
      isError: true,
      result: 'permission denied',
      running: true,
      toolName: 'Bash',
    });

    expect(getStepTitle()).toBe('Running pwd');
    expect(within(screen.getByTestId('assistant-work-tool-call')).getByText('Error')).toHaveClass(
      'text-feedback-danger'
    );
  });

  it('reveals and copies the exact Bash command instead of its description', async () => {
    const user = userEvent.setup();
    await renderStep({
      args: { command: 'pnpm --filter nemo-studio-ui test', description: 'run Studio tests' },
      toolName: 'Bash',
    });

    const step = screen.getByTestId('assistant-work-tool-call');
    expect(getStepTitle()).toBe('Ran run Studio tests');
    expect(step).not.toHaveAttribute('open');

    await user.click(screen.getByText('run Studio tests'));

    expect(step).toHaveAttribute('open');
    expect(within(step).getByText('Command')).toBeVisible();
    expect(within(step).queryByText('Arguments')).not.toBeInTheDocument();
    expect(step).toHaveTextContent('pnpm --filter nemo-studio-ui test');

    await user.click(within(step).getByRole('button', { name: 'Copy' }));

    expect(await navigator.clipboard.readText()).toBe('pnpm --filter nemo-studio-ui test');
    expect(await screen.findByText('Successfully copied to clipboard!')).toBeInTheDocument();
  });

  it('shows the arguments, not an empty Command, for a Bash call without a command', async () => {
    await renderStep({ args: { description: 'check the working directory' }, toolName: 'Bash' });

    const step = screen.getByTestId('assistant-work-tool-call');
    expect(within(step).getByText('Arguments')).toBeInTheDocument();
    expect(within(step).queryByText('Command')).not.toBeInTheDocument();
    expect(step).toHaveTextContent('"description": "check the working directory"');
  });

  it('renders Write as a file change with its line counts and written content', async () => {
    await renderStep({
      args: {
        content: 'export const value = 1;\nexport const next = 2;\n',
        file_path: 'web/packages/studio/src/routes/agents/NewFile.tsx',
      },
      toolName: 'Write',
    });

    const step = screen.getByTestId('assistant-work-file-edit');
    expect(step).toHaveTextContent('Wrote NewFile.tsx');
    expect(within(step).getByTestId('assistant-work-edit-stats')).toHaveTextContent('+2 −0');
    expect(within(step).getByText('+2')).toHaveClass('text-feedback-success');
    expect(within(step).getByText('−0')).toHaveClass('text-feedback-danger');
    expect(
      within(step).getByText('web/packages/studio/src/routes/agents/NewFile.tsx')
    ).toBeInTheDocument();
    expect(step).toHaveTextContent('export const next = 2;');
  });

  it('renders Edit as a file change with its edited line counts', async () => {
    await renderStep({
      args: {
        file_path: 'web/packages/studio/src/routes/agents/ExistingFile.tsx',
        new_string: 'const label = "new";\n',
        old_string: 'const label = "old";\nconst other = 1;\n',
      },
      toolName: 'Edit',
    });

    const step = screen.getByTestId('assistant-work-file-edit');
    expect(step).toHaveTextContent('Edited ExistingFile.tsx');
    expect(within(step).getByTestId('assistant-work-edit-stats')).toHaveTextContent('+1 −2');
  });

  it('shows a running Write with its running action', async () => {
    await renderStep({
      args: { content: 'export const value = 1;\n', file_path: 'src/NewFile.tsx' },
      running: true,
      toolName: 'Write',
    });

    expect(screen.getByTestId('assistant-work-file-edit')).toHaveTextContent('Writing NewFile.tsx');
  });

  it('renders an unknown tool with file-like arguments as a plain Used step', async () => {
    await renderStep({
      args: { file_path: 'src/notes/MEMORY.md', content: 'Pick up milk' },
      toolName: 'save_note',
    });

    expect(screen.queryByTestId('assistant-work-file-edit')).not.toBeInTheDocument();
    expect(screen.getByTestId('assistant-work-tool-call')).toHaveTextContent('Used save_note');
  });

  it('shows a finished failed call as Failed in its collapsed title', async () => {
    await renderStep({
      args: { command: 'pnpm test' },
      isError: true,
      result: 'exit 1',
      toolName: 'Bash',
    });

    expect(getStepTitle()).toBe('Ran pnpm testFailed');
    expect(screen.getByTestId('assistant-work-step-failed')).toHaveClass('text-feedback-danger');
  });

  it('shows the arguments as sent rather than the parsed, rounded args', async () => {
    const user = userEvent.setup();
    await renderStep({
      args: { id: 9007199254740992 },
      argsText: '{"id":9007199254740993}',
      toolName: 'get_job',
    });

    await user.click(screen.getByTestId('nv-collapsible-trigger'));

    expect(screen.getByTestId('assistant-work-tool-call')).toHaveTextContent('9007199254740993');
  });

  it('keeps a failed file write as a tool call so its error shows', async () => {
    await renderStep({
      args: { file_path: 'src/notes/MEMORY.md', content: 'Pick up milk' },
      isError: true,
      result: 'EACCES: permission denied',
      toolName: 'Write',
    });

    expect(screen.queryByTestId('assistant-work-file-edit')).not.toBeInTheDocument();
    const step = screen.getByTestId('assistant-work-tool-call');
    expect(step).toHaveTextContent('Used Write file');
    expect(within(step).getByText('Error')).toBeInTheDocument();
  });
});
