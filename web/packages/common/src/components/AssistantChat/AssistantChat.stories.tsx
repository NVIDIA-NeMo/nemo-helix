// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadMessageLike } from '@assistant-ui/react';
import { AssistantChat } from '@nemo/common/src/components/AssistantChat/index';
import { TooltipProvider } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { http, HttpResponse } from 'msw';
import type { ChatCompletionChunk } from 'openai/resources/index.mjs';

const STORY_INFERENCE_BASE_PATH = '/storybook-inference/v1';
const STORY_COMPLETIONS_API = `${STORY_INFERENCE_BASE_PATH}/chat/completions`;
const STORY_MODEL = 'meta/llama-3.1-8b-instruct';

const getStoryInferenceBaseURL = (): string =>
  typeof window === 'undefined'
    ? `http://localhost${STORY_INFERENCE_BASE_PATH}`
    : `${window.location.origin}${STORY_INFERENCE_BASE_PATH}`;

const initialMessages: readonly ThreadMessageLike[] = [
  {
    role: 'user',
    content: [{ type: 'text', text: 'How should I validate a new chat integration?' }],
  },
  {
    role: 'assistant',
    content: [
      {
        type: 'text',
        text: 'Start with a narrow smoke test: submit a short prompt, confirm a streamed reply renders, then verify stop and reset controls before wiring persistence or feedback.',
      },
    ],
    status: { type: 'complete', reason: 'stop' },
  },
];

const makeChunk = (
  content: string,
  finishReason: ChatCompletionChunk.Choice['finish_reason'] = null
): ChatCompletionChunk => ({
  id: 'story-chat-completion',
  object: 'chat.completion.chunk',
  created: 1740419165,
  model: STORY_MODEL,
  choices: [
    {
      index: 0,
      delta: content ? { content } : {},
      finish_reason: finishReason,
      logprobs: null,
    },
  ],
});

const makeStreamResponse = () => {
  const encoder = new TextEncoder();
  const chunks = [
    'The story uses an MSW-backed inference gateway mock. ',
    'It exercises the same streaming path as the component uses in Studio.',
  ];

  const stream = new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(makeChunk(chunk))}\n\n`));
      });
      controller.enqueue(encoder.encode(`data: ${JSON.stringify(makeChunk('', 'stop'))}\n\n`));
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });

  return new HttpResponse(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
    },
  });
};

interface IndexedStreamOptions {
  readonly chunkCount?: number;
  readonly delayMs?: number;
  readonly hang?: boolean;
}

const makeIndexedExampleResponseHandler = ({
  chunkCount = 6,
  delayMs = 350,
  hang = false,
}: IndexedStreamOptions = {}) => {
  let nextIndex = 0;

  return http.post(STORY_COMPLETIONS_API, () => {
    const encoder = new TextEncoder();
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let requestChunkCount = 0;

    const stream = new ReadableStream({
      start(controller) {
        const enqueueNext = () => {
          requestChunkCount += 1;
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify(
                makeChunk(`${nextIndex++} this is an example response\n`)
              )}\n\n`
            )
          );

          if (!hang && requestChunkCount >= chunkCount) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify(makeChunk('', 'stop'))}\n\n`)
            );
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
            controller.close();
            return;
          }

          timeoutId = setTimeout(enqueueNext, delayMs);
        };

        enqueueNext();
      },
      cancel() {
        if (timeoutId) clearTimeout(timeoutId);
      },
    });

    return new HttpResponse(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
      },
    });
  });
};

const meta = {
  component: AssistantChat,
  title: 'UI/AssistantChat',
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="h-[620px] max-w-[760px]">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  args: {
    model: STORY_MODEL,
    baseURL: getStoryInferenceBaseURL(),
    assistantName: 'Inference Gateway',
    initialMessages,
  },
  parameters: {
    msw: {
      handlers: [http.post(STORY_COMPLETIONS_API, makeStreamResponse)],
    },
  },
} satisfies Meta<typeof AssistantChat>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Empty: Story = {
  args: {
    initialMessages: [],
  },
};

export const IndexedExampleResponses: Story = {
  args: {
    initialMessages: [],
  },
  parameters: {
    msw: {
      handlers: [makeIndexedExampleResponseHandler()],
    },
  },
};

export const HangingExampleResponse: Story = {
  args: {
    initialMessages: [],
  },
  parameters: {
    msw: {
      handlers: [makeIndexedExampleResponseHandler({ delayMs: 500, hang: true })],
    },
  },
};
