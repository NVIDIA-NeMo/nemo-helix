// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AssistantRuntimeProvider } from '@assistant-ui/react';
import { AssistantChatThread as BaseAssistantChatThread } from '@nemo/common/src/components/AssistantChat/AssistantChatThread';
import type { AgentBlockingInputSubmission } from '@studio/components/agents/AgentBlockingInput';
import {
  AgentDecisionInput,
  type AgentDecisionChoice,
} from '@studio/components/agents/AgentDecisionInput';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { AssistantStudioLink } from '@studio/routes/agents/AssistantChatRoute/AssistantStudioLink';
import {
  AssistantToolCallPart,
  CompactAssistantToolCallPart,
} from '@studio/routes/agents/AssistantChatRoute/AssistantToolCallPart';
import { BlockingInputComposer } from '@studio/routes/agents/AssistantChatRoute/BlockingInputComposer';
import type { AssistantChatRuntime } from '@studio/routes/agents/AssistantChatRoute/useAssistantChatRuntime';
import { type FC, useCallback, useLayoutEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router';

const MESSAGE_CONTENT_PROPS = { markdownLinkComponent: AssistantStudioLink };

const EMPTY_STATE = {
  slotHeading: 'Start a NeMo Assistant session',
  slotSubheading: 'Ask NeMo Assistant to work in this workspace.',
};

// Felix's pop-out is a small surface, so it gets a quiet greeting instead of the full empty state.
const COMPACT_EMPTY_STATE = {
  slotHeading: "Hi, I'm Felix",
  slotSubheading: 'What can I help you with?',
  variant: 'compact' as const,
};

const PLACEHOLDER = 'Ask NeMo Assistant to work in this workspace';
const COMPACT_PLACEHOLDER = 'Ask Felix…';

const CHAT_VIEWPORT_SCROLLBAR_CLASS = [
  '[scrollbar-width:thin]',
  '[scrollbar-color:var(--border-color-interaction-base)_transparent]',
  '[&::-webkit-scrollbar]:w-2',
  '[&::-webkit-scrollbar-corner]:bg-transparent',
  '[&::-webkit-scrollbar-track]:bg-transparent',
  '[&::-webkit-scrollbar-thumb]:rounded-full',
  '[&::-webkit-scrollbar-thumb]:bg-[var(--border-color-interaction-base)]',
  '[&::-webkit-scrollbar-thumb:hover]:bg-[var(--border-color-interaction-strong)]',
].join(' ');

interface AssistantChatThreadProps {
  chat: AssistantChatRuntime;
  mode?: 'full' | 'compact';
  onReset?: () => void;
  scrollToBottomSignal?: number;
}

export const AssistantChatThread: FC<AssistantChatThreadProps> = ({
  chat,
  mode = 'full',
  onReset,
  scrollToBottomSignal,
}) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const isCompact = mode === 'compact';
  const chatViewportRef = useRef<HTMLDivElement>(null);
  const {
    decisionChoices,
    decisionRequest,
    decisionStatus,
    handleReset,
    inputRequest,
    inputStatus,
    resolveInputRequest,
    resolveDecisionRequest,
    resolveStudioNavigationRequest,
    runtime,
    skipInputRequest,
    skipDecisionRequest,
    studioNavigationRequest,
    studioNavigationStatus,
  } = chat;

  const studioNavigationChoices = useMemo<readonly AgentDecisionChoice[]>(
    () =>
      studioNavigationRequest
        ? [
            {
              id: 'open-ui',
              label: studioNavigationRequest.suggestion.title,
              description: 'Use the guided Studio UI for this workflow.',
            },
            {
              id: 'continue-chat',
              label: 'Continue in chat',
              description: 'Keep working with NeMo Assistant in this conversation.',
            },
          ]
        : [],
    [studioNavigationRequest]
  );

  const scrollViewportToBottom = useCallback(() => {
    const viewport = chatViewportRef.current;
    if (!viewport) return undefined;

    let secondFrame = 0;
    const firstFrame = window.requestAnimationFrame(() => {
      viewport.scrollTop = viewport.scrollHeight;
      secondFrame = window.requestAnimationFrame(() => {
        viewport.scrollTop = viewport.scrollHeight;
      });
    });

    return () => {
      window.cancelAnimationFrame(firstFrame);
      if (secondFrame) window.cancelAnimationFrame(secondFrame);
    };
  }, []);

  const handleChatReset = useCallback(() => {
    handleReset();
    onReset?.();
  }, [handleReset, onReset]);

  const handleInputSubmit = useCallback(
    async (submission: AgentBlockingInputSubmission) => {
      await resolveInputRequest({
        decision: { value: submission.value },
        displayText: submission.displayText,
      });
    },
    [resolveInputRequest]
  );

  const handleStudioNavigationSubmit = useCallback(
    (choice: AgentDecisionChoice) => {
      const request = studioNavigationRequest;
      if (!request) return;

      if (choice.id === 'open-ui') {
        resolveStudioNavigationRequest('navigate');
        navigate(request.suggestion.href);
        return;
      }

      resolveStudioNavigationRequest('continue');
    },
    [navigate, resolveStudioNavigationRequest, studioNavigationRequest]
  );

  useLayoutEffect(() => {
    if (!studioNavigationRequest && !decisionRequest && !inputRequest) return undefined;
    return scrollViewportToBottom();
  }, [decisionRequest, inputRequest, scrollViewportToBottom, studioNavigationRequest]);

  useLayoutEffect(() => {
    if (scrollToBottomSignal === undefined) return undefined;
    return scrollViewportToBottom();
  }, [scrollToBottomSignal, scrollViewportToBottom]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <BaseAssistantChatThread
        animateAssistantMessageCompletion
        contentClassName={
          mode === 'compact' ? 'w-full px-density-lg' : 'mx-auto w-full max-w-180 px-density-2xl'
        }
        composerContainerClassName={
          mode === 'compact' ? 'w-full px-density-lg' : 'mx-auto w-full max-w-180 px-density-2xl'
        }
        viewportClassName={CHAT_VIEWPORT_SCROLLBAR_CLASS}
        hideAssistantMessageActions
        userMessageVariant="tinted"
        toolCallPartComponent={isCompact ? CompactAssistantToolCallPart : AssistantToolCallPart}
        attributes={{
          ThreadViewport: {
            ref: chatViewportRef,
          },
        }}
        placeholder={isCompact ? COMPACT_PLACEHOLDER : PLACEHOLDER}
        minInputRows={isCompact ? 2 : 3}
        // The compact surface (Felix) already has a "New chat" button in its header.
        hideResetButton={isCompact}
        onReset={handleChatReset}
        showRunningIndicator={!studioNavigationRequest && !decisionRequest && !inputRequest}
        messageContentProps={MESSAGE_CONTENT_PROPS}
        emptyState={isCompact ? COMPACT_EMPTY_STATE : EMPTY_STATE}
        composerOverride={
          studioNavigationRequest ? (
            <AgentDecisionInput
              key={studioNavigationRequest.id}
              request={{
                id: studioNavigationRequest.id,
                title: 'Studio UI available',
                description: `${studioNavigationRequest.suggestion.description} Open it now or continue with NeMo Assistant.`,
              }}
              choices={studioNavigationChoices}
              defaultChoiceId="open-ui"
              status={studioNavigationStatus}
              onSubmit={handleStudioNavigationSubmit}
            />
          ) : decisionRequest ? (
            <AgentDecisionInput
              key={decisionRequest.id}
              request={decisionRequest}
              choices={decisionChoices}
              defaultChoiceId={decisionChoices[0]?.id}
              status={decisionStatus}
              onSubmit={resolveDecisionRequest}
              onSkip={skipDecisionRequest}
            />
          ) : inputRequest ? (
            <BlockingInputComposer
              inputRequest={inputRequest}
              inputStatus={inputStatus}
              workspace={workspace}
              onSubmit={handleInputSubmit}
              onSkip={skipInputRequest}
            />
          ) : undefined
        }
      />
    </AssistantRuntimeProvider>
  );
};
