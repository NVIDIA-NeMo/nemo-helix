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
    className="group/message flex w-full flex-col items-end gap-2 whitespace-normal pl-20"
  >
    <div className="min-w-0 max-w-full rounded-xl rounded-br-none bg-surface-sunken px-4 py-3 dark:bg-surface-raised">
      <AssistantChatMessageContent
        messageContentProps={messageContentProps}
        textSize="lg"
        toolCallPartComponent={toolCallPartComponent}
      />
    </div>
    <div className="flex h-5 shrink-0 items-center">
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
