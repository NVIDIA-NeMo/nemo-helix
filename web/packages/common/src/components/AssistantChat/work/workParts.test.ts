// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_JOB_PROGRESS_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
  ASSISTANT_WORK_DETAILS_LABEL,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import {
  countGroupToolCalls,
  countToolCalls,
  formatToolPayload,
  getWorkedForLabel,
  getWorkGroupLabel,
  getWorkGroupOrdinal,
  groupWorkParts,
} from '@nemo/common/src/components/AssistantChat/work/workParts';

const collapsedDetails = (label: unknown) => ({
  type: 'tool-call',
  toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  args: { label, parts: [] },
});

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

  it('sums the calls of the parts at the given indices', () => {
    const parts = [
      { type: 'tool-call', toolName: 'search' },
      { type: 'text' },
      { type: 'tool-call', toolName: 'search' },
      { type: 'tool-call', toolName: 'fetch' },
    ];

    expect(countGroupToolCalls(parts, [2, 3])).toBe(2);
    expect(countGroupToolCalls(parts, [0, 9])).toBe(1);
  });
});

describe('getWorkGroupOrdinal', () => {
  it('counts the work groups before a part, ignoring text and job progress', () => {
    const parts = [
      { type: 'text' },
      { type: 'reasoning' },
      { type: 'tool-call', toolName: 'search' },
      { type: 'text' },
      { type: 'tool-call', toolName: ASSISTANT_JOB_PROGRESS_TOOL_NAME },
      { type: 'tool-call', toolName: 'search' },
    ];

    expect(getWorkGroupOrdinal(parts, 1)).toBe(0);
    expect(getWorkGroupOrdinal(parts, 5)).toBe(1);
  });
});

describe('getWorkedForLabel', () => {
  it('reads the worked-for label from collapsed work details', () => {
    const parts = [{ type: 'reasoning' }, collapsedDetails(' worked for 42s ')];

    expect(getWorkedForLabel(parts, [0, 1])).toBe('worked for 42s');
    expect(getWorkedForLabel(parts, [0])).toBeUndefined();
  });

  it.each([ASSISTANT_WORK_DETAILS_LABEL, 'Worked for unknown', '  ', 42, undefined])(
    'hides the placeholder label %j',
    (label) => {
      expect(getWorkedForLabel([collapsedDetails(label)], [0])).toBeUndefined();
    }
  );
});

describe('getWorkGroupLabel', () => {
  it('reads Thinking while active, then the count and the time worked', () => {
    expect(getWorkGroupLabel({ active: true, toolCallCount: 2, workedFor: 'worked for 4s' })).toBe(
      'Thinking…'
    );
    expect(getWorkGroupLabel({ active: false, toolCallCount: 3 })).toBe('Ran 3 tool calls');
    expect(getWorkGroupLabel({ active: false, toolCallCount: 1 })).toBe('Ran 1 tool call');
    expect(getWorkGroupLabel({ active: false, toolCallCount: 0 })).toBe('Thought');
    expect(
      getWorkGroupLabel({ active: false, toolCallCount: 3, workedFor: 'worked for 42s' })
    ).toBe('Ran 3 tool calls · worked for 42s');
    expect(getWorkGroupLabel({ active: false, toolCallCount: 0, workedFor: 'worked for 2m' })).toBe(
      'Thought · worked for 2m'
    );
  });
});

describe('formatToolPayload', () => {
  it('pretty-prints JSON values and JSON strings, and keeps other text as is', () => {
    expect(formatToolPayload({ a: 1 })).toEqual({ language: 'json', text: '{\n  "a": 1\n}' });
    expect(formatToolPayload('{"a":1}')).toEqual({ language: 'json', text: '{\n  "a": 1\n}' });
    expect(formatToolPayload('plain output')).toEqual({ language: 'text', text: 'plain output' });
  });

  it.each(['{"id":12345678901234567890}', '{"score":1.0}', '{"n":1e2}'])(
    'keeps %s as sent when parsing would change a number',
    (payload) => {
      expect(formatToolPayload(payload)).toEqual({ language: 'json', text: payload });
    }
  );

  it('pretty-prints JSON whose numbers survive parsing', () => {
    expect(formatToolPayload('{"id":42,"ratio":-0.5,"tag":"v2"}')).toEqual({
      language: 'json',
      text: '{\n  "id": 42,\n  "ratio": -0.5,\n  "tag": "v2"\n}',
    });
  });
});
