// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadMessageLike } from '@assistant-ui/react';
import { humanizeChatError } from '@nemo/common/src/components/AssistantChat/describeChatError';
import { createChatCompletion } from '@nemo/common/src/hooks/useChatCompletion';
import { TooltipProvider } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { http, HttpResponse } from 'msw';
import { useEffect, useState } from 'react';

import { AssistantChat } from './index';

// Walk through what a user sees when an agent chat fails. Each option replays the HTTP response
// the platform returns for that failure through the same OpenAI client and chat runtime Studio
// uses, so the wording comes from `describeChatError.ts` — edit the wording there and
// reload to review it. The code in each label is the model provider's status; the platform
// itself answers HTTP 502 for all of them (the gateway flattens every agent failure).

const STORY_INFERENCE_BASE_PATH = '/storybook-inference/v1';
const STORY_COMPLETIONS_API = `${STORY_INFERENCE_BASE_PATH}/chat/completions`;
const STORY_MODEL = 'story-agent';

const LOREM_IPSUM =
  'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.';

const STORY_BASE_URL = `${window.location.origin}${STORY_INFERENCE_BASE_PATH}`;

interface ErrorScenario {
  readonly key: string;
  readonly label: string;
  readonly status: number;
  /** Undefined means an empty body, as an ingress or proxy error page would return. */
  readonly body?: unknown;
}

// The shape the agents gateway returns: an OpenAI `error` envelope beside FastAPI's `detail`.
const gatewayError = (message: string, type = 'upstream_error') => ({
  error: { message, type, code: null, param: null },
  detail: `Agent returned 502: ${JSON.stringify({ detail: message })}`,
});

// The openai SDK inside the Fabric runtime flattens the model's failure into this text.
const modelError = (providerStatus: number, providerDetail: string) =>
  gatewayError(
    `RuntimeError: internal error: APIStatusError: Error code: ${providerStatus} - ${providerDetail}`
  );

// 410, 424 and 503 provider details are captured from real failures; 404, 401 and 429 are
// representative of the provider's response.
const SCENARIOS: readonly ErrorScenario[] = [
  {
    key: 'model-retired',
    label: '410: model retired',
    status: 502,
    body: modelError(
      410,
      '{\'detail\': \'{"type":"about:blank","title":"Gone","status":410,"detail":"The model \\\'nvidia/nemotron-3-nano-30b-a3b\\\' has reached its end of life on 2026-09-01T09:00:00Z and is no longer available."}\\n\'}'
    ),
  },
  {
    key: 'model-not-found',
    label: '404: model not found',
    status: 502,
    body: modelError(
      404,
      "{'detail': \"No VirtualModel named 'default/missing-model' was found. Inference requests must resolve to a VirtualModel.\"}"
    ),
  },
  {
    key: 'credentials-rejected',
    label: '401: provider rejected credentials',
    status: 502,
    body: modelError(401, "{'detail': 'Incorrect API key provided.'}"),
  },
  {
    key: 'request-rejected',
    label: '424: provider rejected request',
    status: 502,
    body: modelError(
      424,
      "{'detail': \"Model provider 'nvidia-build' at upstream 'https://integrate.api.nvidia.com' rejected the request for model 'default/nvidia-neva-22b' with HTTP status 404 while proxying 'v1/chat/completions'. This is a client-side error and will not resolve by retrying.\"}"
    ),
  },
  {
    key: 'rate-limited',
    label: '429: provider rate limited',
    status: 502,
    body: modelError(429, "{'detail': 'Rate limit exceeded. Try again later.'}"),
  },
  {
    key: 'provider-unavailable',
    label: '503: provider unavailable',
    status: 502,
    body: modelError(
      503,
      "{'detail': 'ResourceExhausted: Worker local total request limit reached (16/16)'}"
    ),
  },
  {
    key: 'runtime-start-failed',
    label: '502: agent runtime failed to start',
    status: 502,
    body: gatewayError(
      'Fabric runtime startup failed: adapter nvidia.fabric.example does not support telemetry.providers value relay'
    ),
  },
  {
    key: 'agent-unresponsive',
    label: '502: agent did not respond',
    status: 502,
    body: gatewayError('Could not connect to agent: ', 'server_error'),
  },
  {
    key: 'no-body',
    label: '502: no response body',
    status: 502,
    body: undefined,
  },
  {
    key: 'unrecognised',
    label: '502: unrecognised error',
    status: 502,
    body: gatewayError('Something no rule knows about.'),
  },
];

// The MSW handler below is registered once per story; it replies with whichever scenario the
// dropdown last selected.
let activeScenario: ErrorScenario = SCENARIOS[0];

const ErrorStatesStory = () => {
  const [scenarioKey, setScenarioKey] = useState(SCENARIOS[0].key);
  const [messages, setMessages] = useState<readonly ThreadMessageLike[]>();
  const scenario = SCENARIOS.find((candidate) => candidate.key === scenarioKey) ?? SCENARIOS[0];

  useEffect(() => {
    let cancelled = false;
    activeScenario = scenario;
    setMessages(undefined);

    const userMessage: ThreadMessageLike = {
      role: 'user',
      content: [{ type: 'text', text: LOREM_IPSUM }],
    };

    // Mirrors the runtime's own failure handling (useAssistantChatRuntime): the human-readable
    // copy becomes the assistant message text and the client's error message its error status.
    createChatCompletion({
      model: STORY_MODEL,
      baseURL: STORY_BASE_URL,
      stream: true,
      messages: [{ role: 'user', content: LOREM_IPSUM }],
    })
      .then(() => 'The mocked endpoint unexpectedly succeeded.')
      .catch((error: unknown) => (error instanceof Error ? error.message : 'Unknown Error'))
      .then((errorMessage) => {
        if (cancelled) return;
        setMessages([
          userMessage,
          {
            role: 'assistant',
            content: [{ type: 'text', text: humanizeChatError(errorMessage) }],
            status: { type: 'incomplete', reason: 'error', error: errorMessage },
          },
        ]);
      });

    return () => {
      cancelled = true;
    };
  }, [scenario]);

  return (
    <div className="flex flex-col gap-density-md">
      <label className="flex items-center gap-density-sm text-sm">
        Error state
        <select
          className="rounded border border-base bg-surface-raised px-2 py-1"
          value={scenarioKey}
          onChange={(event) => setScenarioKey(event.target.value)}
        >
          {SCENARIOS.map((option) => (
            <option key={option.key} value={option.key}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <details className="text-xs text-secondary">
        <summary>API response (HTTP {scenario.status})</summary>
        <pre className="mt-density-xs overflow-auto whitespace-pre-wrap">
          {scenario.body === undefined ? '(empty body)' : JSON.stringify(scenario.body, null, 2)}
        </pre>
      </details>
      <div className="h-[520px]">
        {messages ? (
          <AssistantChat
            key={scenario.key}
            model={STORY_MODEL}
            baseURL={STORY_BASE_URL}
            assistantName="Agent"
            initialMessages={messages}
          />
        ) : null}
      </div>
    </div>
  );
};

const meta = {
  title: 'UI/AssistantChat/Error states',
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-[760px]">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  parameters: {
    msw: {
      handlers: [
        http.post(STORY_COMPLETIONS_API, () =>
          activeScenario.body === undefined
            ? new HttpResponse(null, { status: activeScenario.status })
            : HttpResponse.json(activeScenario.body, { status: activeScenario.status })
        ),
      ],
    },
  },
} satisfies Meta;

export default meta;
type Story = StoryObj<typeof meta>;

export const AgentChatFailures: Story = {
  render: () => <ErrorStatesStory />,
};
