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
import { Flex, Skeleton, Stack, Tooltip } from '@nvidia/foundations-react-core';

// A failed reply keeps a red-barred surface so it doesn't read as a success next to its error banner.
const ASSISTANT_ERROR_SURFACE_CLASS =
  'w-full max-w-full rounded-lg border border-base border-l-4 border-l-feedback-danger bg-surface-base px-density-lg py-density-md shadow ring-1 ring-black/5 dark:ring-white/10';

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
  const failed = useMessageError() !== undefined;
  const messageContent = (
    <Stack align="start" className="w-full min-w-0" gap="density-xs">
      <AssistantChatMessageContent
        contentSurfaceClassName={failed ? ASSISTANT_ERROR_SURFACE_CLASS : undefined}
        messageContentProps={messageContentProps}
        toolCallPartComponent={toolCallPartComponent}
      />
    </Stack>
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
        <Flex align="center" data-testid="assistant-chat-message-actions">
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
        </Flex>
      ) : null}
    </MessagePrimitive.Root>
  );
};
