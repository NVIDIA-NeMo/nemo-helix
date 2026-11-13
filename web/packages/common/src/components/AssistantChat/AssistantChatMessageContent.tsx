// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MessagePrimitive } from '@assistant-ui/react';
import { ChatErrorBanner } from '@nemo/common/src/components/AssistantChat/ChatErrorBanner';
import { CollapsedThinkingToolCall } from '@nemo/common/src/components/AssistantChat/toolCall/CollapsedThinkingToolCall';
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { useIsFailedReplyExplanationPart } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import type { ReactNode } from 'react';

interface AssistantChatMessageContentProps extends MessageRenderProps {
  contentSurfaceClassName?: string;
}

const ContentSurface = ({
  children,
  className,
}: {
  children: ReactNode;
  className: string | undefined;
}) => {
  if (!className) {
    return children;
  }
  return (
    <div className={className} data-testid="assistant-chat-message-surface">
      {children}
    </div>
  );
};

const TextPart = ({
  contentSurfaceClassName,
  messageContentProps,
  text,
}: Pick<AssistantChatMessageContentProps, 'contentSurfaceClassName' | 'messageContentProps'> & {
  text: string;
}) => {
  const isFailureExplanation = useIsFailedReplyExplanationPart();
  if (!text.trim() || isFailureExplanation) {
    return null;
  }
  return (
    <ContentSurface className={contentSurfaceClassName}>
      <MessageContent content={text} {...messageContentProps} />
    </ContentSurface>
  );
};

export const AssistantChatMessageContent = ({
  contentSurfaceClassName,
  messageContentProps,
  toolCallPartComponent,
}: AssistantChatMessageContentProps) => (
  <>
    <MessagePrimitive.Parts
      components={{
        Text: ({ text }) => (
          <TextPart
            contentSurfaceClassName={contentSurfaceClassName}
            messageContentProps={messageContentProps}
            text={text}
          />
        ),
        Image: ({ image, filename }) => (
          <ContentSurface className={contentSurfaceClassName}>
            <img
              src={image}
              alt={filename ?? 'Attached image'}
              className="mt-density-xs max-h-64 w-auto rounded-lg border border-base object-contain"
            />
          </ContentSurface>
        ),
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
