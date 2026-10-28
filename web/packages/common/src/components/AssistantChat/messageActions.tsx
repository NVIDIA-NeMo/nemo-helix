// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ActionBarPrimitive, MessagePrimitive } from '@assistant-ui/react';
import { Check, CopyGeneric } from '@nemo/icons/icons';
import { Tooltip } from '@nvidia/foundations-react-core';

export const ACTION_BUTTON_CLASS =
  'flex cursor-pointer size-7 items-center justify-center rounded border border-transparent text-base bg-transparent text-secondary hover:border-base hover:bg-surface-raised hover:text-primary disabled:cursor-not-allowed disabled:opacity-50';

export const MESSAGE_ACTIONS_CLASS =
  'flex gap-density-xs opacity-0 transition-opacity group-hover/message:opacity-100 group-focus-within/message:opacity-100 [@media(hover:none)]:opacity-100';

export const CopyAction = () => (
  <Tooltip slotContent="Copy message">
    <ActionBarPrimitive.Copy aria-label="Copy message" className={ACTION_BUTTON_CLASS}>
      <MessagePrimitive.If copied>
        <Check size="16" />
      </MessagePrimitive.If>
      <MessagePrimitive.If copied={false}>
        <CopyGeneric size="16" />
      </MessagePrimitive.If>
    </ActionBarPrimitive.Copy>
  </Tooltip>
);
