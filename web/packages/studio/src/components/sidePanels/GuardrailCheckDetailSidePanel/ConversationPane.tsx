// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChatSingle } from '@nemo/icons/icons';
import {
  Accordion,
  CodeSnippet,
  Flex,
  SegmentedControl,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import type { GuardrailCheckEntity } from '@studio/api/guardrail-checks/types';
import { textFromContent } from '@studio/components/dataViews/GuardrailChecksDataView/checkMessages';
import { getHumanReadableFileSize, getTextSizeInBytes } from '@studio/util/files';
import cn from 'classnames';
import { type FC, useState } from 'react';

/** Whether the conversation renders as chat bubbles or as the raw message array. */
type ConversationViewMode = 'chat' | 'json';

const ROLE_LABEL: Record<string, string> = {
  user: 'User',
  assistant: 'Assistant',
  system: 'System',
};

export interface ConversationPaneProps {
  readonly check: GuardrailCheckEntity;
  readonly className?: string;
}

/**
 * Read-only view of the conversation a check runs through its guardrail.
 * Editing lives on the Tests tab; this pane only ever displays.
 *
 * System messages are pulled out into a collapsed accordion — they are
 * routinely far longer than the exchange itself and would otherwise bury it.
 */
export const ConversationPane: FC<ConversationPaneProps> = ({ check, className }) => {
  const [viewMode, setViewMode] = useState<ConversationViewMode>('chat');
  const { messages } = check.data;

  const systemMessages = messages.filter((m) => m.role === 'system');
  const chatMessages = messages.filter((m) => m.role !== 'system');

  return (
    <Stack gap="density-xl" className={cn('overflow-auto p-density-2xl', className)}>
      <Flex align="center" justify="between">
        <Text kind="label/bold/md">Conversation</Text>
        <SegmentedControl
          size="small"
          value={viewMode}
          onValueChange={(v) => setViewMode(v as ConversationViewMode)}
          items={[
            { value: 'chat', children: 'Chat' },
            { value: 'json', children: 'JSON' },
          ]}
        />
      </Flex>

      {viewMode === 'chat' ? (
        <Stack gap="density-md">
          {systemMessages.length > 0 && (
            <Accordion
              items={[
                {
                  value: 'system-prompt',
                  chevronPosition: 'end',
                  slotTrigger: (
                    <Flex align="center" gap="density-sm">
                      <ChatSingle size="16" aria-hidden />
                      <Text kind="label/bold/md">System Prompt</Text>
                      <Text kind="body/regular/md" className="text-secondary">
                        (
                        {getHumanReadableFileSize(
                          getTextSizeInBytes(
                            systemMessages.map((m) => textFromContent(m.content)).join('')
                          )
                        )}
                        )
                      </Text>
                    </Flex>
                  ),
                  slotContent: (
                    <Stack gap="density-sm">
                      {systemMessages.map((m, i) => (
                        <pre key={i} className="whitespace-pre-wrap text-xs font-mono text-primary">
                          {textFromContent(m.content)}
                        </pre>
                      ))}
                    </Stack>
                  ),
                },
              ]}
            />
          )}

          {chatMessages.map((m, i) => {
            const isUser = m.role === 'user';
            return (
              <Stack key={i} gap="density-xs">
                <Text
                  kind="label/bold/xs"
                  className={cn('text-secondary', isUser ? 'text-right' : 'text-left')}
                >
                  {ROLE_LABEL[m.role] ?? m.role}
                </Text>
                {/* Bubbles span the full column; the role is conveyed by the
                    label alignment and fill, not by an indent. */}
                <div
                  className={cn(
                    'rounded-lg p-density-md',
                    isUser ? 'bg-surface-sunken text-right' : 'bg-surface-raised text-left'
                  )}
                >
                  <Text kind="body/regular/md" className="whitespace-pre-wrap">
                    {textFromContent(m.content)}
                  </Text>
                </div>
              </Stack>
            );
          })}
        </Stack>
      ) : (
        <CodeSnippet
          kind="block"
          language="json"
          collapsible
          rows={20}
          value={JSON.stringify(messages, null, 2)}
        />
      )}
    </Stack>
  );
};
