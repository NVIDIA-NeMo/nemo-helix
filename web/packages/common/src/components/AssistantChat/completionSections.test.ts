// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadAssistantMessagePart } from '@assistant-ui/react';
import { appendCompletionDeltaParts } from '@nemo/common/src/components/AssistantChat/completionSections';

const fold = (
  deltas: readonly Parameters<typeof appendCompletionDeltaParts>[1][]
): readonly ThreadAssistantMessagePart[] =>
  deltas.reduce<readonly ThreadAssistantMessagePart[]>(appendCompletionDeltaParts, []);

describe('appendCompletionDeltaParts', () => {
  it('merges consecutive reasoning and text deltas in stream order', () => {
    const parts = fold([
      { reasoning: 'Let me ' },
      { reasoning: 'think.' },
      { text: 'Hello ' },
      { text: 'world' },
    ]);

    expect(parts).toEqual([
      { type: 'reasoning', text: 'Let me think.' },
      { type: 'text', text: 'Hello world' },
    ]);
  });

  it('assembles fragmented tool call arguments by index', () => {
    const parts = fold([
      { text: 'Checking.' },
      {
        toolCalls: [
          { index: 0, id: 'call_a', function: { name: 'get_weather', arguments: '{"ci' } },
          { index: 1, id: 'call_b', function: { name: 'get_time', arguments: '' } },
        ],
      },
      { toolCalls: [{ index: 0, function: { arguments: 'ty":"Paris"}' } }] },
      { toolCalls: [{ index: 1, function: { arguments: '{}' } }] },
    ]);

    expect(parts).toEqual([
      { type: 'text', text: 'Checking.' },
      expect.objectContaining({
        toolCallId: 'call_a',
        toolName: 'get_weather',
        args: { city: 'Paris' },
        argsText: '{"city":"Paris"}',
      }),
      expect.objectContaining({ toolCallId: 'call_b', toolName: 'get_time', args: {} }),
    ]);
  });

  it('keeps empty args while tool arguments are partial JSON', () => {
    const [part] = fold([
      { toolCalls: [{ index: 0, id: 'c', function: { name: 't', arguments: '{"a' } }] },
    ]);

    expect(part).toEqual(expect.objectContaining({ args: {}, argsText: '{"a' }));
  });
});
