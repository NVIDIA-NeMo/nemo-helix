// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MessagePrimitive } from '@assistant-ui/react';
import { ChatErrorBanner } from '@nemo/common/src/components/AssistantChat/ChatErrorBanner';
import { CollapsedThinkingToolCall } from '@nemo/common/src/components/AssistantChat/toolCall/CollapsedThinkingToolCall';
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { useMessageError } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import type { MessageTextSize } from '@nemo/common/src/components/Chat/MessageContent/textSize';

interface AssistantChatMessageContentProps extends MessageRenderProps {
  contentSurfaceClassName?: string;
  textSize?: MessageTextSize;
}

export const AssistantChatMessageContent = ({
  contentSurfaceClassName,
  messageContentProps,
  textSize,
  toolCallPartComponent,
}: AssistantChatMessageContentProps) => {
  // A failed reply's text is shown as the error banner's message, not in the body.
  const failed = useMessageError() !== undefined;

  return (
    <>
      <MessagePrimitive.Parts
        components={{
          Text: ({ text }) => {
            if (failed || !text.trim()) return null;

            const content = (
              <MessageContent content={text} textSize={textSize} {...messageContentProps} />
            );
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
};
