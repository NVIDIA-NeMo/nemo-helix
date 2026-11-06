// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MessagePrimitive } from '@assistant-ui/react';
import { ChatErrorBanner } from '@nemo/common/src/components/AssistantChat/ChatErrorBanner';
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { useMessageError } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { ThoughtStep } from '@nemo/common/src/components/AssistantChat/work/ThoughtStep';
import { ToolCallStep } from '@nemo/common/src/components/AssistantChat/work/ToolCallStep';
import { WorkGroup } from '@nemo/common/src/components/AssistantChat/work/WorkGroup';
import {
  WORK_GROUP_KEY,
  groupWorkParts,
} from '@nemo/common/src/components/AssistantChat/work/workParts';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import type { ReactNode } from 'react';

interface AssistantChatMessageContentProps extends MessageRenderProps {
  contentSurfaceClassName?: string;
}

export const AssistantChatMessageContent = ({
  contentSurfaceClassName,
  messageContentProps,
  toolCallPartComponent: ToolCallPart,
}: AssistantChatMessageContentProps) => {
  // A failed reply's text is shown as the error banner's message, not in the body.
  const failed = useMessageError() !== undefined;
  const surface = (content: ReactNode) =>
    contentSurfaceClassName ? (
      <div className={contentSurfaceClassName} data-testid="assistant-chat-message-surface">
        {content}
      </div>
    ) : (
      content
    );

  return (
    <>
      {/* Adjacent reasoning and tool-call parts coalesce into one collapsed work group. */}
      <MessagePrimitive.GroupedParts groupBy={groupWorkParts}>
        {({ part, children }) => {
          switch (part.type) {
            case WORK_GROUP_KEY:
              return <WorkGroup indices={part.indices}>{children}</WorkGroup>;
            case 'text':
              if (failed || !part.text.trim()) return null;
              return surface(<MessageContent content={part.text} {...messageContentProps} />);
            case 'image':
              return surface(
                <img
                  src={part.image}
                  alt={part.filename ?? 'Attached image'}
                  className="mt-density-xs max-h-64 w-auto rounded-lg border border-base object-contain"
                />
              );
            case 'reasoning':
              return <ThoughtStep text={part.text} />;
            case 'tool-call':
              return ToolCallPart ? (
                <ToolCallPart {...part} />
              ) : (
                <ToolCallStep
                  args={part.args}
                  argsText={part.argsText}
                  isError={part.isError}
                  result={part.result}
                  running={part.status.type === 'running'}
                  toolName={part.toolName}
                />
              );
            default:
              return null;
          }
        }}
      </MessagePrimitive.GroupedParts>
      <MessagePrimitive.Error>
        <ChatErrorBanner />
      </MessagePrimitive.Error>
    </>
  );
};
