// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type MessageSelector,
  messagesPath,
  messagesShape,
} from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';

/** Mirrors ``extractUserFriendlyKeysFromRow``: `null` is a contentless message
 *  it drops, which still consumes an index. */
const selectors = (...roles: (string | null)[]): MessageSelector[] =>
  roles.flatMap((role, index) =>
    role === null ? [] : [{ label: role, role, selector: `conversation[${index}].content` }]
  );

describe('messagesPath', () => {
  it('selects the last turn with the role', () => {
    expect(messagesPath('conversation', 'user')).toBe('conversation[role=user].content');
    expect(messagesPath('conversation', 'assistant')).toBe('conversation[role=assistant].content');
  });
});

describe('messagesShape', () => {
  it.each([
    ['a single-turn exchange', ['user', 'assistant'], { input: true, reference: true }],
    [
      'a multi-turn conversation',
      ['user', 'assistant', 'user', 'assistant'],
      { input: true, reference: true },
    ],
    ['a leading system message', ['system', 'user', 'assistant'], { input: true, reference: true }],
    // The last user turn is unanswered, so the earlier reply is not its reference.
    ['a trailing user turn', ['user', 'assistant', 'user'], { input: true, reference: false }],
    ['no assistant turn', ['user'], { input: true, reference: false }],
    ['a prompts-only file', ['user', 'user'], { input: true, reference: false }],
    ['no user turn', ['system', 'assistant'], { input: false, reference: false }],
    // A contentless turn (e.g. a tool call) between the two does not break the pair.
    [
      'a contentless turn in between',
      ['user', null, 'assistant'],
      { input: true, reference: true },
    ],
    // A contentless assistant turn is not a reference.
    ['a contentless last reply', ['user', null], { input: true, reference: false }],
    ['no messages', [], { input: false, reference: false }],
  ] as const)('handles %s', (_, roles, expected) => {
    expect(messagesShape(selectors(...roles))).toEqual(expected);
  });
});
