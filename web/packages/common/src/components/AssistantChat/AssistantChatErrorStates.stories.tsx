// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadMessageLike } from '@assistant-ui/react';
import { AssistantChat } from '@nemo/common/src/components/AssistantChat';
import { describeChatError } from '@nemo/common/src/components/AssistantChat/describeChatError';
import { getFailedReply } from '@nemo/common/src/components/AssistantChat/messageUtils';
import { TooltipProvider } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';

// Review how an agent chat failure is explained to the user. Pick a failure in the Controls panel
// (or choose "Custom" and paste any error text) to see the reply the chat renders for it: the
// explanation in the red-barred message, the raw error in the red box. The wording comes from
// `describeChatError.ts`, so edit it there and the story updates.
//
// Each option is the error text the OpenAI client surfaces to the chat for that failure. The
// platform answers HTTP 502 for all of them, and the code in each label is the model provider's
// status, which only survives inside the text. The 410, 424 and 503 provider details are captured
// from real failures; 404, 401 and 429 are representative.

const LOREM_IPSUM =
  'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.';

const CUSTOM = 'custom';

const modelError = (providerStatus: number, providerDetail: string) =>
  `RuntimeError: internal error: APIStatusError: Error code: ${providerStatus} - ${providerDetail}`;

const SCENARIOS = {
  'model-retired': {
    label: '410: model retired',
    error: modelError(
      410,
      '{\'detail\': \'{"type":"about:blank","title":"Gone","status":410,"detail":"The model \\\'nvidia/nemotron-3-nano-30b-a3b\\\' has reached its end of life on 2026-09-01T09:00:00Z and is no longer available."}\\n\'}'
    ),
  },
  'model-not-found': {
    label: '404: model not found',
    error: modelError(
      404,
      "{'detail': \"No VirtualModel named 'default/missing-model' was found. Inference requests must resolve to a VirtualModel.\"}"
    ),
  },
  'credentials-rejected': {
    label: '401: provider rejected credentials',
    error: modelError(401, "{'detail': 'Incorrect API key provided.'}"),
  },
  'request-rejected': {
    label: '424: provider rejected request',
    error: modelError(
      424,
      "{'detail': \"Model provider 'nvidia-build' at upstream 'https://integrate.api.nvidia.com' rejected the request for model 'default/nvidia-neva-22b' with HTTP status 404 while proxying 'v1/chat/completions'. This is a client-side error and will not resolve by retrying.\"}"
    ),
  },
  'rate-limited': {
    label: '429: provider rate limited',
    error: modelError(429, "{'detail': 'Rate limit exceeded. Try again later.'}"),
  },
  'provider-unavailable': {
    label: '503: provider unavailable',
    error: modelError(
      503,
      "{'detail': 'ResourceExhausted: Worker local total request limit reached (16/16)'}"
    ),
  },
  'runtime-start-failed': {
    label: '502: agent runtime failed to start',
    error:
      'Fabric runtime startup failed: adapter nvidia.fabric.example does not support telemetry.providers value relay',
  },
  'agent-unresponsive': {
    label: '502: agent did not respond',
    error: 'Could not connect to agent: ',
  },
  'no-body': {
    label: '502: no response body',
    error: '502 status code (no body)',
  },
  unrecognised: {
    label: '502: unrecognised error',
    error: 'Something no rule knows about.',
  },
} as const;

type ScenarioKey = keyof typeof SCENARIOS;

const scenarioLabels = {
  ...Object.fromEntries(Object.entries(SCENARIOS).map(([key, { label }]) => [key, label])),
  [CUSTOM]: 'Custom error text',
};

interface ErrorStatesArgs {
  scenario: ScenarioKey | typeof CUSTOM;
  /** Used when the scenario is "Custom": paste the text the chat received. */
  customError: string;
}

const meta = {
  title: 'UI/AssistantChat/Error states',
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="h-[520px] max-w-[760px]">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  args: {
    scenario: 'model-retired',
    customError: 'RuntimeError: internal error: APIStatusError: Error code: 410 - ...',
  },
  argTypes: {
    scenario: {
      control: { type: 'select', labels: scenarioLabels },
      options: [...Object.keys(SCENARIOS), CUSTOM],
    },
    customError: {
      control: 'text',
      if: { arg: 'scenario', eq: CUSTOM },
    },
  },
  render: ({ scenario, customError }) => {
    const error = scenario === CUSTOM ? customError : SCENARIOS[scenario].error;
    const { parts, status } = getFailedReply(error, describeChatError);
    const messages: readonly ThreadMessageLike[] = [
      { role: 'user', content: [{ type: 'text', text: LOREM_IPSUM }] },
      { role: 'assistant', content: parts, status },
    ];

    return (
      <AssistantChat
        // initialMessages only seeds the thread, so remount when the error changes.
        key={error}
        model="story-agent"
        assistantName="Agent"
        initialMessages={messages}
        describeError={describeChatError}
        disabled
      />
    );
  },
} satisfies Meta<ErrorStatesArgs>;

export default meta;
type Story = StoryObj<typeof meta>;

export const AgentChatFailures: Story = {};
