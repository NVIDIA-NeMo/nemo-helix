// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ActionBarPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
} from '@assistant-ui/react';
import { AssistantChatMessageContent } from '@nemo/common/src/components/AssistantChat/AssistantChatMessageContent';
import { AssistantMessageContentTransition } from '@nemo/common/src/components/AssistantChat/AssistantMessageContentTransition';
import {
  ACTION_BUTTON_CLASS,
  CopyAction,
  MESSAGE_ACTIONS_CLASS,
} from '@nemo/common/src/components/AssistantChat/messageActions';
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { Skeleton, Tooltip } from '@nvidia/foundations-react-core';
import { Refresh } from '@nvidia/react-gui-icons/icons';

const ASSISTANT_MESSAGE_SURFACE_CLASS =
  'w-full max-w-full rounded-lg border border-base border-l-4 border-l-[var(--border-color-brand)] bg-surface-base px-density-lg py-density-md shadow ring-1 ring-black/5 dark:ring-white/10';

export const AssistantMessage = ({
  animateCompletion = false,
  hideAssistantMessageActions,
  messageContentProps,
  showRunningIndicator = true,
  toolCallPartComponent,
}: MessageRenderProps & {
  animateCompletion?: boolean;
  hideAssistantMessageActions?: boolean;
  showRunningIndicator?: boolean;
}) => {
  const hasRenderableContent = useAuiState((state) =>
    state.message.parts.some((part) => part.type !== 'text' || part.text.trim().length > 0)
  );
  const completed = useAuiState((state) => state.message.status?.type === 'complete');
  const messageContent = (
    <AssistantChatMessageContent
      contentSurfaceClassName={ASSISTANT_MESSAGE_SURFACE_CLASS}
      messageContentProps={messageContentProps}
      toolCallPartComponent={toolCallPartComponent}
    />
  );

  return (
    <MessagePrimitive.Root
      data-testid="assistant-chat-message"
      data-testspeaker="assistant"
      className="group/message flex w-full flex-col items-start gap-density-xs whitespace-normal"
    >
      {hasRenderableContent && animateCompletion ? (
        <AssistantMessageContentTransition completed={completed} enabled={animateCompletion}>
          {messageContent}
        </AssistantMessageContentTransition>
      ) : hasRenderableContent ? (
        messageContent
      ) : null}
      {showRunningIndicator ? (
        <MessagePrimitive.If last>
          <ThreadPrimitive.If running>
            <div
              className="flex h-6 w-full items-center"
              data-testid="assistant-chat-running-indicator"
            >
              <Skeleton className="h-density-4 w-full" data-testid="assistant-chat-skeleton" />
            </div>
          </ThreadPrimitive.If>
        </MessagePrimitive.If>
      ) : null}
      {!hideAssistantMessageActions ? (
        <div
          className="flex h-7 items-center pl-density-xs"
          data-testid="assistant-chat-message-actions"
        >
          <ActionBarPrimitive.Root hideWhenRunning className={MESSAGE_ACTIONS_CLASS}>
            <Tooltip slotContent="Regenerate response">
              <ActionBarPrimitive.Reload
                aria-label="Regenerate response"
                className={ACTION_BUTTON_CLASS}
              >
                <Refresh size="16" />
              </ActionBarPrimitive.Reload>
            </Tooltip>
            <CopyAction />
          </ActionBarPrimitive.Root>
        </div>
      ) : null}
    </MessagePrimitive.Root>
  );
};
