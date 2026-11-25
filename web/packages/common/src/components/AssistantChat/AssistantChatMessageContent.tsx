// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MessagePrimitive } from '@assistant-ui/react';
import { ChatErrorBanner } from '@nemo/common/src/components/AssistantChat/ChatErrorBanner';
import type { MessageRenderProps } from '@nemo/common/src/components/AssistantChat/types';
import { useIsFailedReplyExplanationPart } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { ThoughtStep } from '@nemo/common/src/components/AssistantChat/work/ThoughtStep';
import { ToolCallStep } from '@nemo/common/src/components/AssistantChat/work/ToolCallStep';
import { WorkGroup } from '@nemo/common/src/components/AssistantChat/work/WorkGroup';
import {
  groupWorkParts,
  WORK_GROUP_KEY,
} from '@nemo/common/src/components/AssistantChat/work/workParts';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import { type ReactNode, useCallback, useState } from 'react';

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
  toolCallPartComponent: ToolCallPart,
}: AssistantChatMessageContentProps) => {
  // GroupedParts keys groups by position, so reshaping parts on completion remounts them.
  const [workGroupUserOpen, setWorkGroupUserOpen] = useState<ReadonlyMap<number, boolean>>(
    () => new Map()
  );
  const handleWorkGroupUserOpenChange = useCallback((ordinal: number, open: boolean) => {
    setWorkGroupUserOpen((previous) => new Map(previous).set(ordinal, open));
  }, []);

  return (
    <>
      <MessagePrimitive.GroupedParts groupBy={groupWorkParts}>
        {({ part, children }) => {
          switch (part.type) {
            case WORK_GROUP_KEY:
              return (
                <WorkGroup
                  indices={part.indices}
                  onUserOpenChange={handleWorkGroupUserOpenChange}
                  userOpenByOrdinal={workGroupUserOpen}
                >
                  {children}
                </WorkGroup>
              );
            case 'text':
              return (
                <TextPart
                  contentSurfaceClassName={contentSurfaceClassName}
                  messageContentProps={messageContentProps}
                  text={part.text}
                />
              );
            case 'image':
              return (
                <ContentSurface className={contentSurfaceClassName}>
                  <img
                    src={part.image}
                    alt={part.filename ?? 'Attached image'}
                    className="mt-density-xs max-h-64 w-auto rounded-lg border border-base object-contain"
                  />
                </ContentSurface>
              );
            case 'reasoning':
              return <ThoughtStep text={part.text} />;
            case 'tool-call':
              if (ToolCallPart) {
                return <ToolCallPart {...part} />;
              }
              return (
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
