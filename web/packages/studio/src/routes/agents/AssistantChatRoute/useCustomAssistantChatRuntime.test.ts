// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadAssistantMessagePart, ThreadMessageLike } from '@assistant-ui/react';
import { getMessageText } from '@nemo/common/src/components/AssistantChat/messageUtils';
import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import {
  STUDIO_MESSAGE_SUMMARY_END,
  STUDIO_MESSAGE_SUMMARY_START,
} from '@studio/routes/agents/AssistantChatRoute/toolParts';
import {
  type CustomAssistantBeforeRunContext,
  type CustomAssistantRunContext,
  type CustomAssistantRunResult,
  useCustomAssistantChatRuntime,
} from '@studio/routes/agents/AssistantChatRoute/useCustomAssistantChatRuntime';
import { act, renderHook, waitFor } from '@testing-library/react';

const mocks = vi.hoisted(() => ({
  useExternalStoreRuntime: vi.fn((runtime: unknown) => runtime),
}));

vi.mock('@assistant-ui/react', () => ({
  useExternalStoreRuntime: mocks.useExternalStoreRuntime,
}));

interface MockRuntime {
  messages: readonly ThreadMessageLike[];
  onCancel: () => Promise<void>;
}

const getMockRuntime = (runtime: unknown): MockRuntime => {
  if (typeof runtime !== 'object' || runtime === null || !('messages' in runtime)) {
    throw new Error('Expected mocked assistant runtime');
  }

  return runtime as MockRuntime;
};

const getAssistantContent = (messages: readonly ThreadMessageLike[]) => {
  const content = messages[1]?.content;
  if (!Array.isArray(content)) throw new Error('Expected assistant content parts');
  return content;
};

const runContextOf = (onRun: ReturnType<typeof vi.fn>): CustomAssistantRunContext | undefined =>
  onRun.mock.calls[0]?.[0] as CustomAssistantRunContext | undefined;

describe('useCustomAssistantChatRuntime', () => {
  beforeEach(() => {
    mocks.useExternalStoreRuntime.mockClear();
  });

  it('pauses before running and resumes when the before-run hook continues', async () => {
    let continueRun!: () => void;
    const onBeforeRun = vi.fn(async (context: CustomAssistantBeforeRunContext) => {
      context.prepareForUserInput();
      await new Promise<void>((resolve) => {
        continueRun = resolve;
      });
      return 'continue' as const;
    });
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      context.appendAssistantText('Continuing in chat.');
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onBeforeRun, onRun }));

    act(() => {
      void result.current.submitPrompt('Add guardrails');
    });

    await waitFor(() => {
      expect(onBeforeRun).toHaveBeenCalledWith(
        expect.objectContaining({ prompt: 'Add guardrails' })
      );
      expect(onRun).not.toHaveBeenCalled();
      expect(getMockRuntime(result.current.runtime).messages.map(getMessageText)).toEqual([
        'Add guardrails',
      ]);
    });

    await act(async () => {
      continueRun();
    });

    await waitFor(() => {
      expect(onRun).toHaveBeenCalled();
      expect(getMockRuntime(result.current.runtime).messages.map(getMessageText)).toEqual([
        'Add guardrails',
        'Continuing in chat.',
      ]);
    });
  });

  it('does not run when the before-run hook cancels', async () => {
    const onBeforeRun = vi.fn((context: CustomAssistantBeforeRunContext) => {
      context.prepareForUserInput();
      return 'cancel' as const;
    });
    const onRun = vi.fn();
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onBeforeRun, onRun }));

    await act(async () => {
      await result.current.submitPrompt('Open guardrails');
    });

    expect(onRun).not.toHaveBeenCalled();
    expect(getMockRuntime(result.current.runtime).messages.map(getMessageText)).toEqual([
      'Open guardrails',
    ]);
  });

  it('keeps a visible incomplete assistant message when a run fails', async () => {
    const onError = vi.fn();
    const onRun = vi.fn(async () => {
      throw new Error('Connection to NeMo Assistant was interrupted.');
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onError, onRun }));

    await act(async () => {
      await result.current.submitPrompt('Create a guardrail');
    });

    const messages = getMockRuntime(result.current.runtime).messages;
    expect(messages.map(getMessageText)).toEqual([
      'Create a guardrail',
      'NeMo Assistant stopped before completing: Connection to NeMo Assistant was interrupted.',
    ]);
    expect(messages[1]?.status).toEqual({
      type: 'incomplete',
      reason: 'error',
      error: 'Connection to NeMo Assistant was interrupted.',
    });
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Connection to NeMo Assistant was interrupted.' })
    );
  });

  it('preserves streamed assistant content when a run later fails', async () => {
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      context.appendAssistantText('I inspected the current configuration.');
      throw new Error('Connection to NeMo Assistant was interrupted.');
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    await act(async () => {
      await result.current.submitPrompt('Create a guardrail');
    });

    const messages = getMockRuntime(result.current.runtime).messages;
    expect(messages.map(getMessageText)).toEqual([
      'Create a guardrail',
      'I inspected the current configuration.\n\n' +
        'NeMo Assistant stopped before completing: Connection to NeMo Assistant was interrupted.',
    ]);
    expect(messages[1]?.status).toEqual({
      type: 'incomplete',
      reason: 'error',
      error: 'Connection to NeMo Assistant was interrupted.',
    });
  });

  it('shows user interventions between agent messages', async () => {
    let runContext: CustomAssistantRunContext | undefined;
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      runContext = context;
      await new Promise<void>((resolve) => {
        context.signal.addEventListener('abort', () => resolve(), { once: true });
      });
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    act(() => {
      void result.current.submitPrompt('List files');
    });

    await waitFor(() => {
      const runtime = getMockRuntime(result.current.runtime);
      expect(runtime.messages).toHaveLength(2);
      expect(runtime.messages[1]?.role).toBe('assistant');
      expect(runtime.messages[1]?.status?.type).toBe('running');
    });

    act(() => {
      runContext?.appendAssistantText('I need approval first.');
    });

    await waitFor(() => {
      const runtime = getMockRuntime(result.current.runtime);
      expect(runtime.messages.map(getMessageText)).toEqual([
        'List files',
        'I need approval first.',
      ]);
    });

    act(() => {
      runContext?.prepareForUserInput();
      result.current.appendUserMessage('Use rg instead');
      runContext?.appendAssistantText('Using rg instead.');
    });

    await waitFor(() => {
      const runtime = getMockRuntime(result.current.runtime);
      expect(runtime.messages.map((message) => message.role)).toEqual([
        'user',
        'assistant',
        'user',
        'assistant',
      ]);
      expect(runtime.messages.map(getMessageText)).toEqual([
        'List files',
        'I need approval first.',
        'Use rg instead',
        'Using rg instead.',
      ]);
    });

    await act(async () => {
      await getMockRuntime(result.current.runtime).onCancel();
    });
  });

  it('keeps resumed agent output in the same assistant message when no user message was inserted', async () => {
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      context.appendAssistantParts([{ type: 'text', text: 'I need approval first.' }]);
      context.prepareForUserInput();
      context.appendAssistantParts([{ type: 'text', text: '\n\nContinuing after approval.' }]);
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    await act(async () => {
      await result.current.submitPrompt('List files');
    });

    await waitFor(() => {
      const runtime = getMockRuntime(result.current.runtime);
      expect(runtime.messages.map((message) => message.role)).toEqual(['user', 'assistant']);
      expect(runtime.messages.map(getMessageText)).toEqual([
        'List files',
        'I need approval first.\n\nContinuing after approval.',
      ]);
      expect(runtime.messages[1]?.status).toEqual({ type: 'complete', reason: 'stop' });
    });
  });

  it('excludes time waiting for user input from the completed work duration', async () => {
    let now = 1_000;
    const dateNow = vi.spyOn(Date, 'now').mockImplementation(() => now);
    let resumeRun!: () => void;
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      now = 2_000;
      context.appendAssistantParts([{ type: 'text', text: 'I inspected the repository.' }]);
      now = 3_000;
      context.prepareForUserInput();
      await new Promise<void>((resolve) => {
        resumeRun = resolve;
      });
      now = 303_000;
      context.appendAssistantParts([
        {
          type: 'text',
          text: [
            STUDIO_MESSAGE_SUMMARY_START,
            'worked_for: unknown',
            'summary: Finished after receiving the answer.',
            'details_label: worked for unknown',
            STUDIO_MESSAGE_SUMMARY_END,
          ].join('\n'),
        },
      ]);
      now = 305_000;
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    act(() => {
      void result.current.submitPrompt('Inspect the repository');
    });

    await waitFor(() => {
      expect(resumeRun).toBeDefined();
    });

    await act(async () => {
      resumeRun();
    });

    await waitFor(() => {
      expect(getAssistantContent(getMockRuntime(result.current.runtime).messages)).toMatchObject([
        {
          type: 'tool-call',
          toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
          args: { label: 'worked for 4s' },
        },
        { type: 'text', text: 'Finished after receiving the answer.' },
      ]);
    });

    dateNow.mockRestore();
  });

  it('preserves resumed assistant parts when setting assistant text', async () => {
    const bashPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_bash',
      toolName: 'Bash',
      args: { command: 'pwd' },
      argsText: '{"command":"pwd"}',
    };
    let finishRun!: () => void;
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      context.appendAssistantParts([{ type: 'text', text: 'I need approval first.' }, bashPart]);
      context.prepareForUserInput();
      context.setAssistantText('\n\nContinuing after approval.');
      await new Promise<void>((resolve) => {
        finishRun = resolve;
      });
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    act(() => {
      void result.current.submitPrompt('List files');
    });

    await waitFor(() => {
      const runtime = getMockRuntime(result.current.runtime);
      expect(runtime.messages.map((message) => message.role)).toEqual(['user', 'assistant']);
      expect(getAssistantContent(runtime.messages)).toEqual([
        { type: 'text', text: 'I need approval first.' },
        bashPart,
        { type: 'text', text: '\n\nContinuing after approval.' },
      ]);
      expect(runtime.messages[1]?.status).toEqual({ type: 'running' });
    });

    await act(async () => {
      finishRun();
    });

    await waitFor(() => {
      expect(getMockRuntime(result.current.runtime).messages[1]?.status).toEqual({
        type: 'complete',
        reason: 'stop',
      });
    });
  });

  it('combines consecutive subtle tool calls across streamed appends', async () => {
    let runContext: CustomAssistantRunContext | undefined;
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      runContext = context;
      await new Promise<void>((resolve) => {
        context.signal.addEventListener('abort', () => resolve(), { once: true });
      });
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    act(() => {
      void result.current.submitPrompt('Check files');
    });

    await waitFor(() => {
      expect(getMockRuntime(result.current.runtime).messages).toHaveLength(2);
    });

    const bashPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_bash',
      toolName: 'Bash',
      args: { command: 'pwd' },
      argsText: '{"command":"pwd"}',
    };
    const readPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_read',
      toolName: 'Read',
      args: { file_path: 'README.md' },
      argsText: '{"file_path":"README.md"}',
    };

    act(() => {
      runContext?.appendAssistantParts([bashPart]);
    });

    await waitFor(() => {
      const content = getAssistantContent(getMockRuntime(result.current.runtime).messages);
      expect(content).toEqual([bashPart]);
    });

    act(() => {
      runContext?.appendAssistantParts([readPart]);
    });

    await waitFor(() => {
      const content = getAssistantContent(getMockRuntime(result.current.runtime).messages);
      expect(content).toHaveLength(1);
      expect(content[0]).toMatchObject({
        type: 'tool-call',
        toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
        args: {
          actions: [
            { args: { command: 'pwd' }, toolCallId: 'toolu_bash', toolName: 'Bash' },
            { args: { file_path: 'README.md' }, toolCallId: 'toolu_read', toolName: 'Read' },
          ],
        },
      });
    });

    await act(async () => {
      await getMockRuntime(result.current.runtime).onCancel();
    });
  });

  it('does not collapse agent thinking while waiting for user input', async () => {
    const bashPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_bash',
      toolName: 'Bash',
      args: { command: 'pwd' },
      argsText: '{"command":"pwd"}',
    };
    let runContext: CustomAssistantRunContext | undefined;
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      runContext = context;
      context.appendAssistantParts([{ type: 'text', text: 'I will inspect the repo first.' }]);
      context.appendAssistantParts([bashPart]);
      context.prepareForUserInput();
      await new Promise<void>((resolve) => {
        context.signal.addEventListener('abort', () => resolve(), { once: true });
      });
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    act(() => {
      void result.current.submitPrompt('Check files');
    });

    await waitFor(() => {
      expect(runContext).toBeDefined();
      const content = getAssistantContent(getMockRuntime(result.current.runtime).messages);
      expect(content).toEqual([{ type: 'text', text: 'I will inspect the repo first.' }, bashPart]);
      expect(getMockRuntime(result.current.runtime).messages[1]?.status).toEqual({
        type: 'complete',
        reason: 'stop',
      });
    });

    await act(async () => {
      await getMockRuntime(result.current.runtime).onCancel();
    });
  });

  it('keeps completed tool activity before the final summary', async () => {
    const bashPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_bash',
      toolName: 'Bash',
      args: { command: 'pwd' },
      argsText: '{"command":"pwd"}',
    };
    const readPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_read',
      toolName: 'Read',
      args: { file_path: 'README.md' },
      argsText: '{"file_path":"README.md"}',
    };
    let runContext: CustomAssistantRunContext | undefined;
    let finishRun!: () => void;
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      runContext = context;
      context.appendAssistantParts([
        { type: 'text', text: 'I will inspect the repo first.' },
        bashPart,
      ]);
      await new Promise<void>((resolve) => {
        finishRun = resolve;
      });
      context.appendAssistantParts([
        readPart,
        {
          type: 'text',
          text: 'I found the relevant files.\n\nI checked the repo.\n\nTests passed.',
        },
      ]);
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    act(() => {
      void result.current.submitPrompt('Check files');
    });

    await waitFor(() => {
      expect(runContext).toBeDefined();
      const content = getAssistantContent(getMockRuntime(result.current.runtime).messages);
      expect(content).toEqual([{ type: 'text', text: 'I will inspect the repo first.' }, bashPart]);
    });

    await act(async () => {
      finishRun();
    });

    await waitFor(() => {
      const content = getAssistantContent(getMockRuntime(result.current.runtime).messages);
      expect(content).toMatchObject([
        {
          type: 'tool-call',
          toolName: ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
          args: {
            text: 'I will inspect the repo first.\n\nI found the relevant files.',
          },
        },
        {
          type: 'tool-call',
          toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
          args: {
            actions: [
              { toolName: 'Bash', args: { command: 'pwd' } },
              { toolName: 'Read', args: { file_path: 'README.md' } },
            ],
          },
        },
        { type: 'text', text: 'I checked the repo.\n\nTests passed.' },
      ]);
      expect(getMockRuntime(result.current.runtime).messages[1]?.status).toEqual({
        type: 'complete',
        reason: 'stop',
      });
    });
  });
  it('starts a new assistant message for tool calls that follow a picker answer', async () => {
    let runContext: CustomAssistantRunContext | undefined;
    const onRun = vi.fn(async (context: CustomAssistantRunContext) => {
      runContext = context;
      await new Promise<void>((resolve) => {
        context.signal.addEventListener('abort', () => resolve(), { once: true });
      });
    });
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    act(() => {
      void result.current.submitPrompt('Anonymize a dataset');
    });

    await waitFor(() => {
      expect(getMockRuntime(result.current.runtime).messages).toHaveLength(2);
    });

    const pickerPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_select',
      toolName: 'select_dataset_file',
      args: {},
      argsText: '{}',
    };
    const afterPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_after',
      toolName: 'nemo_api',
      args: { resource: 'files' },
      argsText: '{"resource":"files"}',
    };

    act(() => {
      runContext?.appendAssistantParts([pickerPart]);
      // The picker is shown, which closes the active assistant message...
      runContext?.prepareForUserInput();
      // ...but a tool step still streams in before the user answers, which
      // re-attaches the run to that now-closed message.
      runContext?.appendAssistantParts([pickerPart]);
    });

    act(() => {
      result.current.appendUserMessage('Selected dataset: test-pi/titanic.parquet');
      runContext?.appendAssistantParts([afterPart]);
    });

    await waitFor(() => {
      const messages = getMockRuntime(result.current.runtime).messages;
      // The tool call that followed the answer must render below it, not merged
      // back into the assistant message that came before.
      expect(messages.map((message) => message.role)).toEqual([
        'user',
        'assistant',
        'user',
        'assistant',
      ]);
      expect(getMessageText(messages[2]!)).toBe('Selected dataset: test-pi/titanic.parquet');
      expect(JSON.stringify(messages[3]?.content)).toContain('toolu_after');
      expect(JSON.stringify(messages[1]?.content)).not.toContain('toolu_after');
    });

    await act(async () => {
      await getMockRuntime(result.current.runtime).onCancel();
    });
  });
  it('keeps a returned result below a picker answer instead of dropping it', async () => {
    let resolveRun: ((result: CustomAssistantRunResult) => void) | undefined;
    const onRun = vi.fn(
      () =>
        new Promise<CustomAssistantRunResult>((resolve) => {
          resolveRun = resolve;
        })
    );
    const { result } = renderHook(() => useCustomAssistantChatRuntime({ onRun }));

    act(() => {
      void result.current.submitPrompt('Anonymize a dataset');
    });

    await waitFor(() => {
      expect(getMockRuntime(result.current.runtime).messages).toHaveLength(2);
    });

    act(() => {
      runContextOf(onRun)?.appendAssistantText('Picking a dataset.');
      // Showing the picker completes the active assistant message...
      runContextOf(onRun)?.prepareForUserInput();
      // ...and the answer is appended after it.
      result.current.appendUserMessage('Selected dataset: titanic.parquet');
    });

    act(() => {
      resolveRun?.({ text: 'Anonymized 100 rows.' });
    });

    await waitFor(() => {
      const messages = getMockRuntime(result.current.runtime).messages;
      // The result must land in a new message below the answer, not be dropped.
      expect(messages.map((message) => message.role)).toEqual([
        'user',
        'assistant',
        'user',
        'assistant',
      ]);
      expect(getMessageText(messages[3]!)).toBe('Anonymized 100 rows.');
    });
  });
});
