// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  lastExchange,
  type MessageSelector,
} from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';

/** Mirrors ``extractUserFriendlyKeysFromRow``: `null` is a contentless message
 *  it drops, which still consumes an index. */
const selectors = (...roles: (string | null)[]): MessageSelector[] =>
  roles.flatMap((role, index) =>
    role === null ? [] : [{ label: role, role, selector: `conversation[${index}].content` }]
  );

describe('lastExchange', () => {
  it('pairs a single-turn exchange', () => {
    expect(lastExchange(selectors('user', 'assistant'))).toEqual({
      user: 'conversation[0].content',
      assistant: 'conversation[1].content',
    });
  });

  it('pairs the last exchange of a multi-turn conversation', () => {
    expect(lastExchange(selectors('user', 'assistant', 'user', 'assistant'))).toEqual({
      user: 'conversation[2].content',
      assistant: 'conversation[3].content',
    });
  });

  it('ignores a leading system message', () => {
    expect(lastExchange(selectors('system', 'user', 'assistant'))).toEqual({
      user: 'conversation[1].content',
      assistant: 'conversation[2].content',
    });
  });

  // Regression: picking each role independently paired assistant[1] with the
  // later user[2].
  it('ignores a trailing unanswered user turn rather than pairing across it', () => {
    expect(lastExchange(selectors('user', 'assistant', 'user'))).toEqual({
      user: 'conversation[0].content',
      assistant: 'conversation[1].content',
    });
  });

  it('returns the last user turn and no reference when the file has no assistant', () => {
    expect(lastExchange(selectors('user'))).toEqual({
      user: 'conversation[0].content',
      assistant: null,
    });
  });

  it('takes the last user turn of a prompts-only file', () => {
    expect(lastExchange(selectors('user', 'user'))).toEqual({
      user: 'conversation[1].content',
      assistant: null,
    });
  });

  it('rejects both when the assistant turn is not preceded by a user turn', () => {
    expect(lastExchange(selectors('system', 'assistant'))).toEqual({
      user: null,
      assistant: null,
    });
  });

  // Adjacency uses the selector's index, not its position in the list.
  it('rejects when a contentless message sits between the turns', () => {
    expect(lastExchange(selectors('user', null, 'assistant'))).toEqual({
      user: null,
      assistant: null,
    });
  });

  it('returns nothing for an empty selector list', () => {
    expect(lastExchange([])).toEqual({ user: null, assistant: null });
  });
});
