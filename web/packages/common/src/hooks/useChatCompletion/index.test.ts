// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createChatCompletion } from '@nemo/common/src/hooks/useChatCompletion';
import type { ChatCompletion, ChatCompletionChunk } from 'openai/resources/index.mjs';
import type { Stream } from 'openai/streaming.mjs';

const mocks = vi.hoisted(() => ({
  create: vi.fn(),
  openAIConstructor: vi.fn(),
  platformFetch: vi.fn(),
}));

vi.mock('openai', () => ({
  default: class MockOpenAI {
    constructor(options: unknown) {
      mocks.openAIConstructor(options);
    }

    chat = { completions: { create: mocks.create } };
  },
}));

vi.mock('@nemo/sdk/src/utils/platformRequest', () => ({
  platformFetch: mocks.platformFetch,
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
    mocks.openAIConstructor.mockReset();
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

  it('omits authorization headers when no explicit access token is provided', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    mocks.create.mockResolvedValue({ choices: [] });

    await createChatCompletion({
      baseURL: 'http://localhost/no-explicit-auth/v1',
      model: 'default/guarded-model',
      messages: [{ role: 'user', content: 'Tell me about bananas.' }],
      stream: false,
    });

    expect(mocks.create.mock.calls[0]?.[1]?.headers).not.toHaveProperty('Authorization');
    warn.mockRestore();
  });

  it('routes a plain model on its own name', async () => {
    mocks.create.mockResolvedValue({ choices: [] });

    await createChatCompletion({
      workspace: 'plain-ws',
      accessToken: 'test-token',
      model: 'base-model',
      messages: [{ role: 'user', content: 'Tell me about bananas.' }],
      stream: false,
    });

    const { baseURL } = mocks.openAIConstructor.mock.calls[0]?.[0] as { baseURL: string };
    expect(baseURL).toMatch(/\/workspaces\/plain-ws\/model\/base-model\/-\/v1\/?$/);
    expect(mocks.create.mock.calls[0]?.[0]).toMatchObject({ model: 'base-model' });
  });

  it('routes a LoRA adapter through its base model and keeps the composite in the body', async () => {
    mocks.create.mockResolvedValue({ choices: [] });
    const model = 'base-model&adapters/adapter-ws/my-adapter';

    await createChatCompletion({
      workspace: 'route-ws',
      accessToken: 'test-token',
      model,
      messages: [{ role: 'user', content: 'Tell me about bananas.' }],
      stream: false,
    });

    const { baseURL } = mocks.openAIConstructor.mock.calls[0]?.[0] as { baseURL: string };
    expect(baseURL).toMatch(/\/workspaces\/route-ws\/model\/base-model\/-\/v1\/?$/);
    expect(mocks.create.mock.calls[0]?.[0]).toMatchObject({ model });
  });

  it('constructs OpenAI clients with the shared platform fetch boundary', async () => {
    mocks.create.mockResolvedValue({ choices: [] });

    await createChatCompletion({
      baseURL: 'http://localhost/shared-fetch/v1',
      accessToken: 'test-token',
      model: 'default/guarded-model',
      messages: [{ role: 'user', content: 'Tell me about bananas.' }],
      stream: false,
    });

    expect(mocks.openAIConstructor).toHaveBeenCalledWith(
      expect.objectContaining({ fetch: mocks.platformFetch })
    );
  });

  it('sends extra headers without leaking them into the request body', async () => {
    mocks.create.mockReturnValue(Promise.resolve(completion));

    await createChatCompletion({
      baseURL: 'http://localhost/v1',
      accessToken: 'test-token',
      model: 'default/guarded-model',
      messages: [{ role: 'user', content: 'Tell me about the sea.' }],
      extraHeaders: { 'X-Nemo-Session-Id': 'session-1' },
    });

    const [body, options] = mocks.create.mock.calls[0];
    expect(options.headers).toMatchObject({ 'X-Nemo-Session-Id': 'session-1' });
    expect(body).not.toHaveProperty('extraHeaders');
  });
});
