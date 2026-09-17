// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME } from '@studio/routes/agents/AssistantChatRoute/jobProgressConsts';
import {
  getAssistantPartsFromAssistantEvent,
  getAssistantTextFromAssistantEvent,
  parseJsonObject,
  parseSseChunk,
} from '@studio/routes/agents/AssistantChatRoute/stream';
import {
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
} from '@studio/routes/agents/AssistantChatRoute/toolParts';

describe('Assistant stream utilities', () => {
  it('parses SSE events and preserves incomplete trailing data', () => {
    const parsed = parseSseChunk(
      [
        'data: {"type":"assistant"}',
        '',
        'event: custom_event',
        'data: {"request_id":"req-1"}',
        '',
        'event: don',
      ].join('\n')
    );

    expect(parsed.events).toEqual([
      { event: undefined, data: '{"type":"assistant"}' },
      { event: 'custom_event', data: '{"request_id":"req-1"}' },
    ]);
    expect(parsed.rest).toBe('event: don');
  });

  it('extracts assistant text and tool-call parts from Assistant events', () => {
    const event = {
      type: 'assistant',
      message: {
        content: [
          { type: 'text', text: 'I can check that.' },
          {
            type: 'tool_use',
            id: 'toolu_question',
            name: 'AskUserQuestion',
            input: { question: 'Continue?' },
          },
          { type: 'tool_use', id: 'toolu_hidden', name: 'TaskUpdate', input: { status: 'done' } },
          { type: 'tool_use', id: 'toolu_1', name: 'Bash', input: { command: 'pwd' } },
          { type: 'tool_use', id: 'toolu_grep', name: 'Grep', input: { pattern: 'TODO' } },
          { type: 'tool_use', id: 'toolu_2', name: 'Read', input: { file_path: 'README.md' } },
        ],
      },
    };

    expect(getAssistantPartsFromAssistantEvent(event)).toEqual([
      { type: 'text', text: 'I can check that.' },
      {
        type: 'tool-call',
        toolCallId: 'assistant-subtle-tools-toolu_question-toolu_2',
        toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
        args: {
          actions: [
            {
              args: { question: 'Continue?' },
              toolCallId: 'toolu_question',
              toolName: 'AskUserQuestion',
            },
            { args: { status: 'done' }, toolCallId: 'toolu_hidden', toolName: 'TaskUpdate' },
            { args: { command: 'pwd' }, toolCallId: 'toolu_1', toolName: 'Bash' },
            { args: { pattern: 'TODO' }, toolCallId: 'toolu_grep', toolName: 'Grep' },
            { args: { file_path: 'README.md' }, toolCallId: 'toolu_2', toolName: 'Read' },
          ],
        },
        argsText:
          '{"actions":[{"args":{"question":"Continue?"},"toolCallId":"toolu_question","toolName":"AskUserQuestion"},{"args":{"status":"done"},"toolCallId":"toolu_hidden","toolName":"TaskUpdate"},{"args":{"command":"pwd"},"toolCallId":"toolu_1","toolName":"Bash"},{"args":{"pattern":"TODO"},"toolCallId":"toolu_grep","toolName":"Grep"},{"args":{"file_path":"README.md"},"toolCallId":"toolu_2","toolName":"Read"}]}',
      },
    ]);
    expect(getAssistantTextFromAssistantEvent(event)).toBe('I can check that.');
  });

  it('preserves subtle Assistant tool calls from streamed parts', () => {
    const event = {
      type: 'assistant',
      message: {
        content: [
          { type: 'tool_use', id: 'toolu_1', name: 'Bash', input: { command: 'pwd' } },
          { type: 'tool_use', id: 'toolu_2', name: 'TaskUpdate', input: { status: 'done' } },
          { type: 'tool_use', id: 'toolu_3', name: 'Grep', input: { pattern: 'TODO' } },
          { type: 'tool_use', id: 'toolu_find', name: 'FindFiles', input: { query: 'TODO' } },
          { type: 'tool_use', id: 'toolu_task', name: 'TaskCreate', input: { task: 'check' } },
          { type: 'tool_use', id: 'toolu_search', name: 'ToolSearch', input: { query: 'read' } },
          {
            type: 'tool_use',
            id: 'toolu_4',
            name: 'AskUserQuestion',
            input: { question: 'Continue?' },
          },
        ],
      },
    };

    expect(getAssistantPartsFromAssistantEvent(event)).toEqual([
      {
        type: 'tool-call',
        toolCallId: 'assistant-subtle-tools-toolu_1-toolu_4',
        toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
        args: {
          actions: [
            { args: { command: 'pwd' }, toolCallId: 'toolu_1', toolName: 'Bash' },
            { args: { status: 'done' }, toolCallId: 'toolu_2', toolName: 'TaskUpdate' },
            { args: { pattern: 'TODO' }, toolCallId: 'toolu_3', toolName: 'Grep' },
            { args: { query: 'TODO' }, toolCallId: 'toolu_find', toolName: 'FindFiles' },
            { args: { task: 'check' }, toolCallId: 'toolu_task', toolName: 'TaskCreate' },
            { args: { query: 'read' }, toolCallId: 'toolu_search', toolName: 'ToolSearch' },
            { args: { question: 'Continue?' }, toolCallId: 'toolu_4', toolName: 'AskUserQuestion' },
          ],
        },
        argsText:
          '{"actions":[{"args":{"command":"pwd"},"toolCallId":"toolu_1","toolName":"Bash"},{"args":{"status":"done"},"toolCallId":"toolu_2","toolName":"TaskUpdate"},{"args":{"pattern":"TODO"},"toolCallId":"toolu_3","toolName":"Grep"},{"args":{"query":"TODO"},"toolCallId":"toolu_find","toolName":"FindFiles"},{"args":{"task":"check"},"toolCallId":"toolu_task","toolName":"TaskCreate"},{"args":{"query":"read"},"toolCallId":"toolu_search","toolName":"ToolSearch"},{"args":{"question":"Continue?"},"toolCallId":"toolu_4","toolName":"AskUserQuestion"}]}',
      },
    ]);
  });

  it('groups unknown non-file-change tool calls with other subtle streamed parts', () => {
    const event = {
      type: 'assistant',
      message: {
        content: [
          {
            type: 'tool_use',
            id: 'toolu_inspect',
            name: 'InspectWorkspace',
            input: { query: 'symbols' },
          },
          { type: 'tool_use', id: 'toolu_read', name: 'Read', input: { file_path: 'README.md' } },
        ],
      },
    };

    expect(getAssistantPartsFromAssistantEvent(event)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
        args: {
          actions: [
            {
              args: { query: 'symbols' },
              toolCallId: 'toolu_inspect',
              toolName: 'InspectWorkspace',
            },
            { args: { file_path: 'README.md' }, toolCallId: 'toolu_read', toolName: 'Read' },
          ],
        },
      },
    ]);
  });

  it('keeps the job progress card out of subtle streamed tool groups', () => {
    const event = {
      type: 'assistant',
      message: {
        content: [
          { type: 'tool_use', id: 'toolu_bash', name: 'Bash', input: { command: 'pwd' } },
          {
            type: 'tool_use',
            id: 'toolu_job',
            name: ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME,
            input: { job_name: 'studio-job-1' },
          },
          { type: 'tool_use', id: 'toolu_read', name: 'Read', input: { file_path: 'README.md' } },
        ],
      },
    };

    expect(getAssistantPartsFromAssistantEvent(event)).toEqual([
      {
        type: 'tool-call',
        toolCallId: 'toolu_bash',
        toolName: 'Bash',
        args: { command: 'pwd' },
        argsText: '{"command":"pwd"}',
      },
      {
        type: 'tool-call',
        toolCallId: 'toolu_job',
        toolName: ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME,
        args: { job_name: 'studio-job-1' },
        argsText: '{"job_name":"studio-job-1"}',
      },
      {
        type: 'tool-call',
        toolCallId: 'toolu_read',
        toolName: 'Read',
        args: { file_path: 'README.md' },
        argsText: '{"file_path":"README.md"}',
      },
    ]);
  });

  it('combines subtle streamed tool calls across whitespace-only text parts', () => {
    const event = {
      type: 'assistant',
      message: {
        content: [
          { type: 'tool_use', id: 'toolu_bash', name: 'Bash', input: { command: 'pwd' } },
          { type: 'text', text: '\n\n' },
          { type: 'tool_use', id: 'toolu_glob', name: 'Glob', input: { pattern: '*' } },
        ],
      },
    };

    expect(getAssistantPartsFromAssistantEvent(event)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
        args: {
          actions: [
            { args: { command: 'pwd' }, toolCallId: 'toolu_bash', toolName: 'Bash' },
            { args: { pattern: '*' }, toolCallId: 'toolu_glob', toolName: 'Glob' },
          ],
        },
      },
    ]);
  });

  it('scopes fallback streamed tool-call ids to the Assistant message id', () => {
    const event = {
      type: 'assistant',
      message: {
        id: 'msg_123',
        content: [{ type: 'tool_use', name: 'Read', input: { file_path: 'README.md' } }],
      },
    };

    expect(getAssistantPartsFromAssistantEvent(event)).toEqual([
      {
        type: 'tool-call',
        toolCallId: 'assistant-tool-msg_123-Read-0',
        toolName: 'Read',
        args: { file_path: 'README.md' },
        argsText: '{"file_path":"README.md"}',
      },
    ]);
  });

  it('returns undefined and logs when JSON parsing fails', () => {
    const loggerSpy = vi.spyOn(logger, 'error').mockImplementation(() => undefined);

    expect(parseJsonObject('{')).toBeUndefined();
    expect(loggerSpy).toHaveBeenCalledWith(
      expect.stringContaining('Failed to parse NeMo Assistant stream JSON'),
      expect.any(Error)
    );

    loggerSpy.mockRestore();
  });
  it('maps a reasoning part to a collapsible thinking part', () => {
    const parts = getAssistantPartsFromAssistantEvent({
      type: 'assistant',
      message: {
        id: 'msg-reasoning',
        content: [{ type: 'reasoning', text: 'The user asked a math question.' }],
      },
    });

    expect(parts).toEqual([
      {
        type: 'tool-call',
        toolCallId: 'assistant-thinking-msg-reasoning-0',
        toolName: ASSISTANT_THINKING_TOOL_NAME,
        args: { text: 'The user asked a math question.' },
        argsText: JSON.stringify({ text: 'The user asked a math question.' }),
      },
    ]);
  });

  it('drops an empty reasoning part', () => {
    const parts = getAssistantPartsFromAssistantEvent({
      type: 'assistant',
      message: { id: 'msg-empty', content: [{ type: 'reasoning', text: '' }] },
    });

    expect(parts).toEqual([]);
  });

  it('drops a whitespace-only reasoning part', () => {
    const parts = getAssistantPartsFromAssistantEvent({
      type: 'assistant',
      message: { id: 'msg-whitespace', content: [{ type: 'reasoning', text: '   \n  ' }] },
    });

    expect(parts).toEqual([]);
  });
});
