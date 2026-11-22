// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadMessageLike } from '@assistant-ui/react';
import { AssistantChat } from '@nemo/common/src/components/AssistantChat/index';
import { Anchor, TooltipProvider } from '@nvidia/foundations-react-core';
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

const RICH_RESPONSE = `# Fine-tuning plan for your support agent

Here's how I'd improve the agent's answers about **billing disputes**. The short version: the base model is fine, but it needs *domain examples* and a ***stricter refusal policy*** for account changes.

## 1. Prepare the dataset

Collect around **2,000** resolved tickets and convert them to chat format. Keep these fields:

- \`messages\`: the full conversation, with the agent's final reply as the last turn
- \`category\`: one of \`billing\`, \`refund\`, or \`account\`
- \`resolved\`: *only keep tickets marked* **true**
  - Drop tickets that were escalated to a human
  - Drop tickets shorter than two turns

See the [dataset format guide](https://docs.nvidia.com/nemo/microservices/latest/fine-tune/format-training-dataset.html) for the full schema.

## 2. Launch the job

1. Upload the dataset to a fileset.
2. Start a LoRA job with a low learning rate.
3. Evaluate against the held-out set before deploying.

\`\`\`bash
nemo files upload ./tickets.jsonl --fileset support-tickets
nemo customization jobs create --config lora.yaml --dataset support-tickets
\`\`\`

The config only needs a few overrides:

\`\`\`python
config = {
    "model": "nvidia/nemotron-3-super",
    "training": {"type": "lora", "epochs": 3, "learning_rate": 1e-4},
    "dataset": {"fileset": "support-tickets", "validation_split": 0.1},
}
\`\`\`

### What to watch

> If validation loss rises after the first epoch, stop early. The agent will start repeating ticket boilerplate.

Compare the tuned agent against the baseline on **accuracy**, *tone*, and refusal rate. You can read more about evaluation metrics in the [NeMo Evaluator docs](https://docs.nvidia.com/nemo/evaluator/latest/), or ask me to set up the comparison for you.

---

Want me to draft \`lora.yaml\` next?`;

const RICH_RESPONSE_MESSAGES: readonly ThreadMessageLike[] = [
  {
    role: 'user',
    content: [
      { type: 'text', text: 'How should I fine-tune my support agent for billing questions?' },
    ],
  },
  {
    role: 'assistant',
    content: [{ type: 'text', text: RICH_RESPONSE }],
    status: { type: 'complete', reason: 'stop' },
  },
];

export const RichFormatting: Story = {
  args: {
    initialMessages: RICH_RESPONSE_MESSAGES,
  },
};

export const RichFormattingWithLinks: Story = {
  args: {
    initialMessages: RICH_RESPONSE_MESSAGES,
    messageContentProps: {
      markdownLinkComponent: ({ href, children }) => (
        <Anchor href={href} external>
          {children}
        </Anchor>
      ),
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
