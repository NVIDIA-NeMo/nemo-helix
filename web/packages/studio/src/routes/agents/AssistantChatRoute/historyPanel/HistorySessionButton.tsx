// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { ToolCallSummary } from '@studio/routes/agents/AssistantChatRoute/historyPanel/ArtifactSections';
import {
  getCompactRelativeTime,
  getHistorySessionTitle,
} from '@studio/routes/agents/AssistantChatRoute/historyPanel/helpers';
import type { AssistantHistorySession } from '@studio/routes/agents/AssistantChatRoute/types';
import cn from 'classnames';
import { MessageSquare, Trash2 } from 'lucide-react';
import React from 'react';

interface HistorySessionButtonProps {
  active: boolean;
  onDelete: () => void;
  onSelect: () => void;
  session: AssistantHistorySession;
}

export const HistorySessionButton = ({
  active,
  onDelete,
  onSelect,
  session,
}: HistorySessionButtonProps): React.JSX.Element => {
  const sessionTitle = getHistorySessionTitle(session);
  const timestamp = new Date(session.mtime * 1000).toLocaleString();
  const prompt = session.first_prompt.trim();
  const tooltip = prompt ? `${timestamp}\n\n${prompt}` : timestamp;

  return (
    <Flex
      align="center"
      className={cn(
        'w-full border-b border-base pr-density-sm transition-colors hover:bg-surface-sunken',
        active && 'bg-surface-sunken'
      )}
    >
      <Button
        kind="tertiary"
        type="button"
        aria-current={active ? 'page' : undefined}
        aria-label={`Open chat ${sessionTitle}`}
        title={tooltip}
        className="h-auto min-w-0 flex-1 justify-start rounded-none px-density-md py-density-sm text-left"
        onClick={onSelect}
      >
        <Stack gap="density-xs">
          <Flex align="center" gap="density-sm">
            <span
              className={cn(
                'flex size-6 shrink-0 items-center justify-center text-secondary',
                active && 'text-accent'
              )}
            >
              <MessageSquare size={12} />
            </span>
            <Flex align="center" justify="between" gap="density-sm" className="min-w-0 flex-1">
              <Text kind="body/regular/sm" className="min-w-0 flex-1 line-clamp-2">
                {sessionTitle}
              </Text>
              <Text kind="body/regular/sm" className="shrink-0 whitespace-nowrap text-secondary">
                {getCompactRelativeTime(session.mtime)}
              </Text>
            </Flex>
          </Flex>
          {session.tool_calls.length > 0 && (
            <div className="pl-8">
              <ToolCallSummary toolCalls={session.tool_calls} />
            </div>
          )}
        </Stack>
      </Button>
      <Tooltip slotContent={`Delete ${sessionTitle}`} side="left">
        <Button
          aria-label={`Delete chat ${sessionTitle}`}
          color="danger"
          kind="tertiary"
          size="small"
          type="button"
          onClick={onDelete}
        >
          <Trash2 size={16} />
        </Button>
      </Tooltip>
    </Flex>
  );
};
