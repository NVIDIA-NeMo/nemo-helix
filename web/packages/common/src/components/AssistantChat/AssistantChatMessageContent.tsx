// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MessagePrimitive } from '@assistant-ui/react';
import { ReasoningPart } from '@nemo/common/src/components/AssistantChat/ReasoningPart';
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import { Banner } from '@nvidia/foundations-react-core';

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
        Reasoning: ReasoningPart,
        tools: { Fallback: toolCallPartComponent },
      }}
    />
    <MessagePrimitive.Error>
      <Banner kind="inline" status="error" className="mt-density-sm">
        There was an error generating a response.
      </Banner>
    </MessagePrimitive.Error>
  </>
);
