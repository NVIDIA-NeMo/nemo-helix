// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MessagePrimitive, useAuiState } from '@assistant-ui/react';
import { describeChatError } from '@nemo/common/src/components/AssistantChat/describeChatError';
import { CollapsedThinkingToolCall } from '@nemo/common/src/components/AssistantChat/toolCall/CollapsedThinkingToolCall';
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import { Banner } from '@nvidia/foundations-react-core';

// The raw failure text is already shown as the message body; the banner adds what it means and
// what to do when a rule recognises it, and falls back to the generic line otherwise.
const ChatErrorBanner = () => {
  const error = useAuiState((s) => {
    const status = s.message.status;
    return status?.type === 'incomplete' && status.reason === 'error'
      ? String(status.error ?? '')
      : '';
  });
  const described = describeChatError(error);

  return (
    <Banner kind="inline" status="error" className="mt-density-sm">
      {described
        ? `${described.title} ${described.hint}`
        : 'There was an error generating a response.'}
    </Banner>
  );
};

interface AssistantChatMessageContentProps extends MessageRenderProps {
  contentSurfaceClassName?: string;
}

export const AssistantChatMessageContent = ({
  contentSurfaceClassName,
  messageContentProps,
  toolCallPartComponent,
}: AssistantChatMessageContentProps) => (
  <>
    <MessagePrimitive.Parts
      components={{
        Text: ({ text }) => {
          if (!text.trim()) return null;

          const content = <MessageContent content={text} {...messageContentProps} />;
          return contentSurfaceClassName ? (
            <div className={contentSurfaceClassName} data-testid="assistant-chat-message-surface">
              {content}
            </div>
          ) : (
            content
          );
        },
        Image: ({ image, filename }) => {
          const content = (
            <img
              src={image}
              alt={filename ?? 'Attached image'}
              className="mt-density-xs max-h-64 w-auto rounded-lg border border-base object-contain"
            />
          );
          return contentSurfaceClassName ? (
            <div className={contentSurfaceClassName} data-testid="assistant-chat-message-surface">
              {content}
            </div>
          ) : (
            content
          );
        },
        Reasoning: ({ status, text }) => (
          <CollapsedThinkingToolCall
            label={status.type === 'running' ? 'Thinking' : 'Thought process'}
            text={text}
          />
        ),
        tools: { Fallback: toolCallPartComponent },
      }}
    />
    <MessagePrimitive.Error>
      <ChatErrorBanner />
    </MessagePrimitive.Error>
  </>
);
