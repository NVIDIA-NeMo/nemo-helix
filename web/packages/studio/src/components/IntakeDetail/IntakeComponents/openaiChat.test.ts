// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toChatMessages } from '@studio/components/IntakeDetail/IntakeComponents/openaiChat';

describe('toChatMessages', () => {
  it('reads a request transcript', () => {
    expect(
      toChatMessages({
        model: 'gpt-4o',
        messages: [
          { role: 'system', content: 'Be brief.' },
          { role: 'user', content: 'Hello' },
        ],
      })
    ).toEqual([
      expect.objectContaining({ role: 'system', content: 'Be brief.' }),
      expect.objectContaining({ role: 'user', content: 'Hello' }),
    ]);
  });

  it('reads a request recorded as an HTTP body', () => {
    // Some observers store the whole request, putting the body under `content`.
    expect(
      toChatMessages({
        headers: { authorization: 'redacted' },
        content: { messages: [{ role: 'user', content: 'Hello' }] },
      })
    ).toEqual([expect.objectContaining({ role: 'user', content: 'Hello' })]);
  });

  it('reads a bare array of messages', () => {
    expect(toChatMessages([{ role: 'user', content: 'Hello' }])).toEqual([
      expect.objectContaining({ role: 'user', content: 'Hello' }),
    ]);
  });

  it('assumes the assistant for a choice that omits its role', () => {
    expect(
      toChatMessages({
        choices: [
          { finish_reason: 'stop', message: { content: 'Hi', reasoning_content: 'Greet them.' } },
        ],
      })
    ).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'Hi',
        reasoning: 'Greet them.',
        finishReason: 'stop',
      }),
    ]);
  });

  it('reads a single assistant message response', () => {
    expect(
      toChatMessages({
        assistant_message: { role: 'assistant', content: 'Done', tool_calls: [] },
        finish_reason: 'stop',
      })
    ).toEqual([expect.objectContaining({ role: 'assistant', content: 'Done' })]);
  });

  it('joins a recorded request and its reply into one conversation', () => {
    const messages = toChatMessages({
      content: { messages: [{ role: 'user', content: 'Hello' }] },
      choices: [{ message: { role: 'assistant', content: 'Hi' } }],
    });

    expect(messages?.map((message) => message.role)).toEqual(['user', 'assistant']);
  });

  it('keeps tool calls and the id a tool result answers', () => {
    const messages = toChatMessages({
      messages: [
        {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call-1',
              type: 'function',
              function: { name: 'search', arguments: '{"q":"a"}' },
            },
          ],
        },
        { role: 'tool', tool_call_id: 'call-1', content: '{"hits":0}' },
      ],
    });

    expect(messages?.[0].toolCalls).toEqual([
      { id: 'call-1', name: 'search', arguments: '{"q":"a"}' },
    ]);
    expect(messages?.[1]).toMatchObject({ role: 'tool', toolCallId: 'call-1' });
  });

  it('flattens multi-part content and names the parts that carry no text', () => {
    const messages = toChatMessages({
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Describe this' },
            { type: 'image_url', image_url: { url: 'https://example.test/a.png' } },
          ],
        },
      ],
    });

    expect(messages?.[0].content).toBe('Describe this\n\n[image_url]');
  });

  it('rejects a payload whose `message` is prose rather than a message object', () => {
    expect(toChatMessages({ message: 'I could not find a booking system.' })).toBeNull();
  });

  it('rejects an array that is not entirely messages', () => {
    expect(toChatMessages([{ role: 'user', content: 'Hello' }, { step: 1 }])).toBeNull();
  });

  it('rejects payloads with no conversation in them', () => {
    expect(toChatMessages({ error: { type: 'APIError', message: 'overloaded' } })).toBeNull();
    expect(toChatMessages({ messages: [] })).toBeNull();
    expect(toChatMessages('plain text')).toBeNull();
    expect(toChatMessages(null)).toBeNull();
  });
});
