// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ActionBarPrimitive, MessagePrimitive } from '@assistant-ui/react';
import { Check, CopyGeneric } from '@nemo/icons/icons';
import { Button, Tooltip } from '@nvidia/foundations-react-core';
import type { ComponentProps } from 'react';

export const MESSAGE_ACTION_ICON_SIZE = '12';

export const MESSAGE_ACTIONS_CLASS =
  'flex gap-density-xxs opacity-0 transition-opacity group-hover/message:opacity-100 group-focus-within/message:opacity-100 [@media(hover:none)]:opacity-100';

/** The tiny tertiary icon button used in a message's hover action row. */
export const MessageActionButton = (props: ComponentProps<typeof Button>) => (
  <Button kind="tertiary" size="tiny" type="button" {...props} />
);

export const CopyAction = () => (
  <Tooltip slotContent="Copy message">
    <ActionBarPrimitive.Copy asChild>
      <MessageActionButton aria-label="Copy message">
        <MessagePrimitive.If copied>
          <Check size={MESSAGE_ACTION_ICON_SIZE} />
        </MessagePrimitive.If>
        <MessagePrimitive.If copied={false}>
          <CopyGeneric size={MESSAGE_ACTION_ICON_SIZE} />
        </MessagePrimitive.If>
      </MessageActionButton>
    </ActionBarPrimitive.Copy>
  </Tooltip>
);
