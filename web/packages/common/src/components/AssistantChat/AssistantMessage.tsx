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
  CopyAction,
  MESSAGE_ACTION_ICON_SIZE,
  MESSAGE_ACTIONS_CLASS,
  MessageActionButton,
} from '@nemo/common/src/components/AssistantChat/messageActions';
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { useMessageError } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { Retry } from '@nemo/icons/icons';
import { Skeleton, Tooltip } from '@nvidia/foundations-react-core';

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
  const failed = useMessageError() !== undefined;
  // A failed reply always renders so its error banner shows, even with no text.
  const hasRenderableContent =
    useAuiState((state) =>
      state.message.parts.some((part) => part.type !== 'text' || part.text.trim().length > 0)
    ) || failed;
  const completed = useAuiState((state) => state.message.status?.type === 'complete');
  const messageContent = (
    <div className="flex w-full min-w-0 flex-col items-start gap-density-xs">
      <AssistantChatMessageContent
        contentSurfaceClassName="w-full min-w-0 max-w-full"
        messageContentProps={messageContentProps}
        toolCallPartComponent={toolCallPartComponent}
      />
    </div>
  );

  return (
    <MessagePrimitive.Root
      data-testid="assistant-chat-message"
      data-testspeaker="assistant"
      className="group/message flex w-full flex-col items-start gap-density-xl whitespace-normal"
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
        <div className="flex items-center" data-testid="assistant-chat-message-actions">
          <ActionBarPrimitive.Root hideWhenRunning className={MESSAGE_ACTIONS_CLASS}>
            <CopyAction />
            <Tooltip slotContent="Regenerate response">
              <ActionBarPrimitive.Reload asChild>
                <MessageActionButton aria-label="Regenerate response">
                  <Retry size={MESSAGE_ACTION_ICON_SIZE} />
                </MessageActionButton>
              </ActionBarPrimitive.Reload>
            </Tooltip>
          </ActionBarPrimitive.Root>
        </div>
      ) : null}
    </MessagePrimitive.Root>
  );
};
