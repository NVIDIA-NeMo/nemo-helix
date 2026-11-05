// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  countLines,
  countToolCalls,
  formatToolPayload,
  formatWorkDuration,
  getFileEdit,
  getFileLanguage,
  getWorkGroupLabel,
  getWorkGroupOrdinal,
  groupWorkParts,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/workParts';
import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_JOB_PROGRESS_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
} from '@nemo/common/src/components/AssistantChat/toolParts';

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

describe('getWorkGroupOrdinal', () => {
  it('numbers each run of work parts in order', () => {
    const parts = [
      { type: 'reasoning' },
      { type: 'tool-call', toolName: 'a' },
      { type: 'text' },
      { type: 'tool-call', toolName: 'b' },
    ];
    expect(getWorkGroupOrdinal(parts, 0)).toBe(0);
    expect(getWorkGroupOrdinal(parts, 3)).toBe(1);
  });
});

describe('getWorkGroupLabel', () => {
  it('reads Thinking while running, then the count and duration', () => {
    expect(getWorkGroupLabel({ running: true, toolCallCount: 2, durationMs: 4000 })).toBe(
      'Thinking…'
    );
    expect(getWorkGroupLabel({ running: false, toolCallCount: 3, durationMs: 12_400 })).toBe(
      'Ran 3 tool calls · 12s'
    );
    expect(getWorkGroupLabel({ running: false, toolCallCount: 1 })).toBe('Ran 1 tool call');
    expect(getWorkGroupLabel({ running: false, toolCallCount: 0, durationMs: 300 })).toBe(
      'Thought · <1s'
    );
  });
});

describe('formatWorkDuration', () => {
  it('formats seconds and minutes', () => {
    expect(formatWorkDuration(59_000)).toBe('59s');
    expect(formatWorkDuration(60_000)).toBe('1m');
    expect(formatWorkDuration(72_000)).toBe('1m 12s');
  });
});

describe('getFileEdit', () => {
  it('detects a written file with its line count', () => {
    expect(getFileEdit({ file_path: 'notes/MEMORY.md', content: 'milk\neggs\n' })).toEqual({
      path: 'notes/MEMORY.md',
      content: 'milk\neggs\n',
      additions: 2,
      deletions: 0,
    });
    expect(getFileEdit({ path: 'a.ts' })).toBeUndefined();
    expect(getFileEdit({ query: 'weather' })).toBeUndefined();
  });

  it('keeps the replaced and replacement text of an edit', () => {
    expect(getFileEdit({ path: 'a.ts', old_string: 'x\ny', new_string: 'z' })).toEqual({
      path: 'a.ts',
      before: 'x\ny',
      after: 'z',
      additions: 1,
      deletions: 2,
    });
  });

  it('sums the line counts of a multi-edit', () => {
    expect(
      getFileEdit({
        file_path: 'a.ts',
        edits: [
          { old_string: 'a', new_string: 'b\nc' },
          { old_string: '', new_string: 'd' },
        ],
      })
    ).toEqual({ path: 'a.ts', additions: 3, deletions: 1 });
  });

  it('counts lines without a trailing newline', () => {
    expect(countLines('')).toBe(0);
    expect(countLines('one')).toBe(1);
    expect(countLines('one\r\ntwo\r\n')).toBe(2);
  });

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
