// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createChatCompletion } from '@nemo/common/src/hooks/useChatCompletion';
import type { ChatCompletion, ChatCompletionChunk } from 'openai/resources/index.mjs';
import type { Stream } from 'openai/streaming.mjs';

const mocks = vi.hoisted(() => ({
  create: vi.fn(),
}));

vi.mock('openai', () => ({
  default: class MockOpenAI {
    chat = { completions: { create: mocks.create } };
  },
}));

const completion: ChatCompletion = {
  id: 'chatcmpl-blocked',
  object: 'chat.completion',
  created: 1,
  model: 'default/guarded-model',
  choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: "I'm sorry, I can't respond to that.",
        refusal: null,
      },
      finish_reason: 'content_filter',
      logprobs: null,
    },
  ],
};

describe('createChatCompletion', () => {
  beforeEach(() => {
    mocks.create.mockReset();
  });

  it('returns an immediate JSON completion when a streamed request is blocked', async () => {
    const response = new Response(JSON.stringify(completion), {
      headers: { 'content-type': 'application/json' },
    });
    mocks.create.mockReturnValue({
      asResponse: vi.fn().mockResolvedValue(response),
    });

    const result = await createChatCompletion({
      baseURL: 'http://localhost/v1',
      accessToken: 'test-token',
      model: 'default/guarded-model',
      messages: [{ role: 'user', content: 'Tell me about bananas.' }],
      stream: true,
    });

    expect(result).toEqual(completion);
  });

  it('preserves an SSE stream for an ordinary streamed completion', async () => {
    const stream = {
      controller: new AbortController(),
      async *[Symbol.asyncIterator]() {},
    } as unknown as Stream<ChatCompletionChunk>;
    const request = Promise.resolve(stream);
    Object.assign(request, {
      asResponse: vi
        .fn()
        .mockResolvedValue(
          new Response(null, { headers: { 'content-type': 'text/event-stream' } })
        ),
    });
    mocks.create.mockReturnValue(request);

    const result = await createChatCompletion({
      baseURL: 'http://localhost/v1',
      accessToken: 'test-token',
      model: 'default/guarded-model',
      messages: [{ role: 'user', content: 'Tell me about the moon.' }],
      stream: true,
    });

    expect(result).toBe(stream);
  });
});
