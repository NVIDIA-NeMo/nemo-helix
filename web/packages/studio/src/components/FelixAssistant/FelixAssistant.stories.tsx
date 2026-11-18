// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadMessageLike } from '@assistant-ui/react';
import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import { Text, TooltipProvider } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { FelixAssistant } from '@studio/components/FelixAssistant';
import { AssistantChatProvider } from '@studio/routes/agents/AssistantChatRoute/context/AssistantChatProvider';
import {
  AssistantChatContext,
  useAssistantChatContext,
} from '@studio/routes/agents/AssistantChatRoute/context/useAssistantChatContext';
import type { AssistantInputRequest } from '@studio/routes/agents/AssistantChatRoute/types';
import { type FC, type ReactNode, useEffect, useMemo, useState } from 'react';
import { Route, Routes } from 'react-router';
import { userEvent, within } from 'storybook/test';

const AWAITING_INPUT_REQUEST: AssistantInputRequest = {
  requestId: 'story-request',
  kind: 'dataset_file',
  input: {},
};

/** A reply whose last part is a tool call still in flight, which is what poses Felix. */
const messagesWithToolInFlight = (
  toolName: string,
  args: Record<string, string>
): readonly ThreadMessageLike[] => [
  { role: 'user', content: 'Please get this done.' },
  {
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'story-tool-call',
        toolName,
        args,
        argsText: JSON.stringify(args),
      },
    ],
  },
];

const ACTIVITY_MESSAGES = {
  // Creating something on the platform: Felix hammers.
  building: messagesWithToolInFlight('nemo_api', {
    resource: 'guardrail.configs',
    action: 'create',
  }),
  // Looking something up: Felix reads his book.
  docs: messagesWithToolInFlight('Read', { file_path: 'skills/guardrails/SKILL.md' }),
} as const;

const RUNNING_MESSAGES: readonly ThreadMessageLike[] = [
  { role: 'user', content: 'Please get this done.' },
];

const FINISHED_MESSAGES: readonly ThreadMessageLike[] = [
  { role: 'user', content: 'Please get this done.' },
  {
    role: 'assistant',
    content: [{ type: 'text', text: 'All done!' }],
    status: { type: 'complete', reason: 'stop' },
  },
];

/** In the `Done` story the finished state lasts this long before the fake run starts again. */
const FINISHED_REST_MS = 6_000;

interface ChatStateOverrideProps {
  isRunning: boolean;
  awaitingInput: boolean;
  activity?: keyof typeof ACTIVITY_MESSAGES;
  /** Runs a fake run on a loop: working for this long, then finishing successfully. */
  finishAfterMs?: number;
  children: ReactNode;
}

/**
 * Re-provides the real chat context with a forced run state, so each story can show Felix's
 * thinking, needs-you and work poses without a backend. Everything else (composer, history) is the
 * real runtime; sending a message will fail here because no assistant agent is running.
 */
const ChatStateOverride: FC<ChatStateOverrideProps> = ({
  isRunning,
  awaitingInput,
  activity,
  finishAfterMs,
  children,
}) => {
  const context = useAssistantChatContext();
  const [cycleFinished, setCycleFinished] = useState(false);

  // For the Done story: work for a while, finish successfully, rest, and go round again, so the
  // celebration can be watched repeatedly.
  useEffect(() => {
    if (finishAfterMs === undefined) return;
    const timer = setTimeout(
      () => setCycleFinished((finished) => !finished),
      cycleFinished ? FINISHED_REST_MS : finishAfterMs
    );
    return () => clearTimeout(timer);
  }, [cycleFinished, finishAfterMs]);

  const isLooping = finishAfterMs !== undefined;
  const value = useMemo(() => {
    const loopMessages = cycleFinished ? FINISHED_MESSAGES : RUNNING_MESSAGES;
    return {
      ...context,
      chat: {
        ...context.chat,
        isRunning: isLooping
          ? !cycleFinished
          : isRunning || !!activity || awaitingInput || context.chat.isRunning,
        inputRequest: awaitingInput ? AWAITING_INPUT_REQUEST : context.chat.inputRequest,
        messages: isLooping
          ? loopMessages
          : activity
            ? ACTIVITY_MESSAGES[activity]
            : context.chat.messages,
      },
    };
  }, [activity, awaitingInput, context, cycleFinished, isLooping, isRunning]);

  return <AssistantChatContext.Provider value={value}>{children}</AssistantChatContext.Provider>;
};

/** Stand-in page content so you can see how Felix floats over a real-looking page. */
const PagePlaceholder: FC = () => (
  <div className="flex min-h-screen flex-col gap-density-lg bg-surface-sunken p-density-2xl text-primary">
    <Text kind="body/bold/2xl">Workspace page</Text>
    <Text kind="body/regular/md" color="secondary">
      Felix starts in the bottom-right corner and stays put while the page scrolls underneath him.
      Drag him to move him (or focus him and use the arrow keys); double-click to put him back.
    </Text>
    {Array.from({ length: 6 }, (_, index) => (
      <div key={index} className="rounded-lg border border-base bg-surface-raised p-density-lg">
        <Text kind="label/bold/md">Resource {index + 1}</Text>
      </div>
    ))}
  </div>
);

interface FelixStageProps {
  isRunning?: boolean;
  awaitingInput?: boolean;
  activity?: keyof typeof ACTIVITY_MESSAGES;
  finishAfterMs?: number;
}

const FelixStage: FC<FelixStageProps> = ({
  isRunning = false,
  awaitingInput = false,
  activity,
  finishAfterMs,
}) => (
  <AssistantChatProvider workspace="default">
    <ChatStateOverride
      isRunning={isRunning}
      awaitingInput={awaitingInput}
      activity={activity}
      finishAfterMs={finishAfterMs}
    >
      <PagePlaceholder />
      <FelixAssistant />
    </ChatStateOverride>
  </AssistantChatProvider>
);

const meta: Meta<typeof FelixStage> = {
  component: FelixStage,
  title: 'Assistant/FelixAssistant',
  decorators: [
    // The chat thread reads :workspace from the URL, so render under a matching route.
    (Story) => (
      <Routes>
        <Route path="/workspaces/:workspace/*" element={<Story />} />
      </Routes>
    ),
    (Story) => (
      <TooltipProvider>
        <ToastProvider>
          <Story />
        </ToastProvider>
      </TooltipProvider>
    ),
  ],
  parameters: {
    layout: 'fullscreen',
    router: { initialPath: '/workspaces/default/jobs' },
    // Felix is position: fixed, so render each story in its own iframe in the docs page.
    docs: { story: { inline: false, iframeHeight: 720 } },
  },
};

export default meta;

type Story = StoryObj<typeof FelixStage>;

export const Closed: Story = {};

export const Thinking: Story = {
  args: { isRunning: true },
};

export const NeedsYourInput: Story = {
  args: { awaitingInput: true },
};

/** The assistant is creating something: Felix swings his hammer. */
export const Building: Story = {
  args: { activity: 'building' },
};

/** The assistant is looking something up: Felix reads his book. */
export const ReadingDocs: Story = {
  args: { activity: 'docs' },
};

/**
 * A run finishes successfully: Felix throws his hands up and hops for 3 seconds, then goes back to
 * normal. The fake run loops (1.5s of work, 6s at rest) so you can watch it more than once.
 */
export const Done: Story = {
  args: { finishAfterMs: 1_500 },
};

export const Open: Story = {
  play: async ({ canvasElement }) => {
    await userEvent.click(
      await within(canvasElement).findByRole('button', { name: 'Open NeMo Assistant chat' })
    );
  },
};
