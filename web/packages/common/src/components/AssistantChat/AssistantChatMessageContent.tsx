// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MessagePrimitive } from '@assistant-ui/react';
import { CollapsedThinkingToolCall } from '@nemo/common/src/components/AssistantChat/toolCall/CollapsedThinkingToolCall';
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { useMessageError } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import { Banner } from '@nvidia/foundations-react-core';

// The message body above carries the human-readable explanation; the banner shows the raw error
// (status and provider detail) for anyone who needs to look closer.
const ChatErrorBanner = () => {
  const error = useMessageError();

  return (
    <Banner kind="inline" status="error" className="mt-density-sm">
      {error}
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
