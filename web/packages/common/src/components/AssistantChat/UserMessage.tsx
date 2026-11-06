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
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { Pencil } from '@nemo/icons/icons';
import { Tooltip } from '@nvidia/foundations-react-core';

export const UserMessage = ({ messageContentProps, toolCallPartComponent }: MessageRenderProps) => (
  <MessagePrimitive.Root
    data-testid="assistant-chat-message"
    data-testspeaker="user"
    className="group/message flex w-full flex-col items-end gap-density-md whitespace-normal pl-density-5xl"
  >
    {/* Prompts read at body/lg where replies use MessageContent's body/md. */}
    <div className="min-w-0 max-w-full rounded-xl rounded-br-none bg-surface-raised px-density-xl py-density-lg [&_li]:text-body-regular-lg [&_p]:text-body-regular-lg">
      <AssistantChatMessageContent
        messageContentProps={messageContentProps}
        toolCallPartComponent={toolCallPartComponent}
      />
    </div>
    <div className="flex shrink-0 items-center">
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
    </div>
  </MessagePrimitive.Root>
);
