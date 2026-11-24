// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ActionBarPrimitive, MessagePrimitive } from '@assistant-ui/react';
import { AssistantChatMessageContent } from '@nemo/common/src/components/AssistantChat/AssistantChatMessageContent';
import {
  CopyAction,
  MESSAGE_ACTION_ICON_SIZE,
  MESSAGE_ACTIONS_CLASS,
  MessageActionButton,
} from '@nemo/common/src/components/AssistantChat/messageActions';
import type {
  MessageRenderProps,
  UserMessageVariant,
} from '@nemo/common/src/components/AssistantChat/types';
import { Pencil } from '@nemo/icons/icons';
import { Flex, Tooltip } from '@nvidia/foundations-react-core';

/** Shape, fill and padding of the user's message, shared by the message and its edit box. */
export const USER_MESSAGE_SURFACE_CLASS: Record<UserMessageVariant, string> = {
  bubble: 'rounded-xl rounded-br-none bg-surface-raised px-density-xl py-density-lg',
  // A light brand wash that adapts to light and dark; text keeps the assistant's size and formatting.
  tinted: 'rounded-xl bg-brand/10 px-density-lg py-density-md',
};

// The original bubble sets its text larger than the assistant's; the tinted variant does not.
const USER_MESSAGE_TEXT_CLASS: Record<UserMessageVariant, string> = {
  bubble: '[&_li]:text-body-regular-lg [&_p]:text-body-regular-lg',
  tinted: '',
};

export const UserMessage = ({
  messageContentProps,
  toolCallPartComponent,
  variant = 'bubble',
}: MessageRenderProps & { variant?: UserMessageVariant }) => (
  <MessagePrimitive.Root
    data-testid="assistant-chat-message"
    data-testspeaker="user"
    className="group/message flex w-full flex-col items-end gap-density-md whitespace-normal pl-density-5xl"
  >
    <div
      className={`min-w-0 max-w-full ${USER_MESSAGE_SURFACE_CLASS[variant]} ${USER_MESSAGE_TEXT_CLASS[variant]}`}
      data-testid="assistant-chat-user-message-surface"
      data-variant={variant}
    >
      <AssistantChatMessageContent
        messageContentProps={messageContentProps}
        toolCallPartComponent={toolCallPartComponent}
      />
    </div>
    <Flex align="center" className="shrink-0">
      <ActionBarPrimitive.Root hideWhenRunning className={MESSAGE_ACTIONS_CLASS}>
        <Tooltip slotContent="Edit message">
          <ActionBarPrimitive.Edit asChild>
            <MessageActionButton aria-label="Edit message">
              <Pencil size={MESSAGE_ACTION_ICON_SIZE} />
            </MessageActionButton>
          </ActionBarPrimitive.Edit>
        </Tooltip>
        <CopyAction />
      </ActionBarPrimitive.Root>
    </Flex>
  </MessagePrimitive.Root>
);
