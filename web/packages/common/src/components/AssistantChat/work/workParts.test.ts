// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_JOB_PROGRESS_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import {
  countToolCalls,
  formatToolPayload,
  getFileLanguage,
  getWorkGroupLabel,
  groupWorkParts,
} from '@nemo/common/src/components/AssistantChat/work/workParts';

describe('groupWorkParts', () => {
  it('groups reasoning and tool calls but leaves text and job progress ungrouped', () => {
    expect(groupWorkParts({ type: 'reasoning' })).toEqual(['group-work']);
    expect(groupWorkParts({ type: 'tool-call', toolName: 'search' })).toEqual(['group-work']);
    expect(groupWorkParts({ type: 'text' })).toBeNull();
    expect(
      groupWorkParts({ type: 'tool-call', toolName: ASSISTANT_JOB_PROGRESS_TOOL_NAME })
    ).toBeNull();
  });
});

describe('countToolCalls', () => {
  it('counts real calls and unpacks synthetic NeMo Assistant parts', () => {
    expect(countToolCalls({ type: 'reasoning' })).toBe(0);
    expect(countToolCalls({ type: 'tool-call', toolName: 'search' })).toBe(1);
    expect(countToolCalls({ type: 'tool-call', toolName: ASSISTANT_THINKING_TOOL_NAME })).toBe(0);
    expect(
      countToolCalls({
        type: 'tool-call',
        toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
        args: { actions: [{}, {}, {}] },
      })
    ).toBe(3);
    expect(
      countToolCalls({
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
        args: {
          parts: [
            { type: 'reasoning', text: 'Plan.' },
            { type: 'tool-call', toolName: 'Read', args: {} },
            {
              type: 'tool-call',
              toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
              args: { actions: [{}, {}] },
            },
          ],
        },
      })
    ).toBe(3);
  });
});

describe('getWorkGroupLabel', () => {
  it('reads Thinking while running, then the count', () => {
    expect(getWorkGroupLabel({ running: true, toolCallCount: 2 })).toBe('Thinking…');
    expect(getWorkGroupLabel({ running: false, toolCallCount: 3 })).toBe('Ran 3 tool calls');
    expect(getWorkGroupLabel({ running: false, toolCallCount: 1 })).toBe('Ran 1 tool call');
    expect(getWorkGroupLabel({ running: false, toolCallCount: 0 })).toBe('Thought');
  });
});

describe('getFileLanguage', () => {
  it('picks a highlighter language from the extension', () => {
    expect(getFileLanguage('src/app.tsx')).toBe('tsx');
    expect(getFileLanguage('MEMORY.md')).toBe('markdown');
    expect(getFileLanguage('Makefile')).toBe('text');
  });
});

describe('formatToolPayload', () => {
  it('pretty-prints JSON values and JSON strings, and keeps other text as is', () => {
    expect(formatToolPayload({ a: 1 })).toEqual({ language: 'json', text: '{\n  "a": 1\n}' });
    expect(formatToolPayload('{"a":1}')).toEqual({ language: 'json', text: '{\n  "a": 1\n}' });
    expect(formatToolPayload('plain output')).toEqual({ language: 'text', text: 'plain output' });
  });
});
