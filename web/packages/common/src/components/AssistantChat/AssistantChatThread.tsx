// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ThreadPrimitive } from '@assistant-ui/react';
import { AssistantComposer } from '@nemo/common/src/components/AssistantChat/AssistantComposer';
import { AssistantMessage } from '@nemo/common/src/components/AssistantChat/AssistantMessage';
import {
  ComposerMode,
  type AssistantChatThreadProps,
} from '@nemo/common/src/components/AssistantChat/types';
import { UserEditComposer } from '@nemo/common/src/components/AssistantChat/UserEditComposer';
import { UserMessage } from '@nemo/common/src/components/AssistantChat/UserMessage';
import { ChatEmptyState } from '@nemo/common/src/components/Chat/ChatEmptyState';
import { Flex, Stack } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { useCallback, useMemo } from 'react';

export type { AssistantChatThreadAttributes } from '@nemo/common/src/components/AssistantChat/types';

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

export const AssistantChatThread = ({
  disabled,
  placeholder,
  onReset,
  showRunningIndicator = true,
  attributes,
  composerMode = ComposerMode.PER_PANEL,
  slotComposerStart,
  emptyState,
  contentClassName,
  composerContainerClassName,
  hideAssistantMessageActions,
  toolCallPartComponent,
  viewportClassName,
  composerOverride,
  messageContentProps,
  enableImageAttachments,
  minInputRows,
  animateAssistantMessageCompletion,
}: AssistantChatThreadProps) => {
  const { className: threadViewportClassName, ...threadViewportAttributes } =
    attributes?.ThreadViewport ?? {};
  const AssistantMessageWithToolCallPart = useCallback(
    () => (
      <AssistantMessage
        animateCompletion={animateAssistantMessageCompletion}
        hideAssistantMessageActions={hideAssistantMessageActions}
        messageContentProps={messageContentProps}
        showRunningIndicator={showRunningIndicator}
        toolCallPartComponent={toolCallPartComponent}
      />
    ),
    [
      animateAssistantMessageCompletion,
      hideAssistantMessageActions,
      messageContentProps,
      showRunningIndicator,
      toolCallPartComponent,
    ]
  );
  const UserMessageWithToolCallPart = useCallback(
    () => (
      <UserMessage
        messageContentProps={messageContentProps}
        toolCallPartComponent={toolCallPartComponent}
      />
    ),
    [messageContentProps, toolCallPartComponent]
  );
  const UserEditComposerWithAttachments = useCallback(
    () => <UserEditComposer enableImageAttachments={enableImageAttachments} />,
    [enableImageAttachments]
  );
  const messageComponents = useMemo(
    () => ({
      AssistantMessage: AssistantMessageWithToolCallPart,
      UserMessage: UserMessageWithToolCallPart,
      UserEditComposer: UserEditComposerWithAttachments,
      SystemMessage: AssistantMessageWithToolCallPart,
    }),
    [AssistantMessageWithToolCallPart, UserMessageWithToolCallPart, UserEditComposerWithAttachments]
  );

  return (
    <ThreadPrimitive.Root className="flex h-full w-full flex-col" role="log">
      <div className="relative min-h-0 flex-1">
        <ThreadPrimitive.Viewport
          {...threadViewportAttributes}
          data-testid="assistant-chat-viewport"
          className={cn(
            'flex h-full min-h-0 flex-col overflow-y-auto',
            CHAT_VIEWPORT_SCROLLBAR_CLASS,
            viewportClassName,
            threadViewportClassName
          )}
        >
          <Stack
            gap="density-lg"
            className={cn('min-h-full w-full py-density-md', contentClassName)}
          >
            <ThreadPrimitive.Empty>
              <ChatEmptyState
                className="h-full min-h-[250px] w-full"
                slotHeading={emptyState?.slotHeading}
                slotSubheading={emptyState?.slotSubheading}
                slotAction={emptyState?.slotAction}
              />
            </ThreadPrimitive.Empty>
            <ThreadPrimitive.Messages components={messageComponents} />
          </Stack>
        </ThreadPrimitive.Viewport>
        <ThreadPrimitive.ScrollToBottom className="absolute bottom-density-sm left-1/2 z-10 -translate-x-1/2 rounded border border-base bg-surface-raised px-density-sm py-density-xs text-sm shadow disabled:hidden">
          Scroll to bottom
        </ThreadPrimitive.ScrollToBottom>
      </div>
      {composerMode !== ComposerMode.BROADCAST_ALL && (
        <Flex
          className={cn('w-full pt-density-lg', composerContainerClassName)}
          data-testid="assistant-chat-composer-container"
        >
          {composerOverride ?? (
            <AssistantComposer
              disabled={disabled}
              placeholder={placeholder}
              onReset={onReset}
              slotComposerStart={slotComposerStart}
              enableImageAttachments={enableImageAttachments}
              minInputRows={minInputRows}
            />
          )}
        </Flex>
      )}
    </ThreadPrimitive.Root>
  );
};
