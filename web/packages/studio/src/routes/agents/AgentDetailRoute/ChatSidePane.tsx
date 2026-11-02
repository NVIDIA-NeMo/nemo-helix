// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import type { FC, ReactNode } from 'react';

interface ChatSidePaneProps {
  title: string;
  /** Which edge of the chat the pane sits on; its border faces the chat. */
  side: 'start' | 'end';
  slotAction?: ReactNode;
  className?: string;
  children: ReactNode;
}

/** A fixed-width pane flanking the Chat tab's conversation, in the Data Designer editor's style. */
export const ChatSidePane: FC<ChatSidePaneProps> = ({
  title,
  side,
  slotAction,
  className,
  children,
}) => (
  <Stack
    role="complementary"
    aria-label={title}
    gap="density-lg"
    className={cn(
      'h-full min-h-0 overflow-y-auto border-base p-density-lg',
      side === 'start' ? 'border-r' : 'border-l',
      className
    )}
  >
    <Flex align="center" justify="between" gap="density-sm" className="min-h-8">
      <Text kind="label/semibold/md">{title}</Text>
      {slotAction}
    </Flex>
    {children}
  </Stack>
);
