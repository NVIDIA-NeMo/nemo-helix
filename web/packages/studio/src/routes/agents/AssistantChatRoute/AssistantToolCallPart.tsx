// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ToolCallMessagePartComponent } from '@assistant-ui/react';
import { RepeatedToolCallsStep } from '@nemo/common/src/components/AssistantChat/chainOfThought/RepeatedToolCallsStep';
import { ThoughtStep } from '@nemo/common/src/components/AssistantChat/chainOfThought/ThoughtStep';
import { ToolCallStep } from '@nemo/common/src/components/AssistantChat/chainOfThought/ToolCallStep';
import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
  isAssistantJobProgressToolName,
  toAssistantToolArgs,
  type AssistantToolArgs,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import { AssistantStudioLink } from '@studio/routes/agents/AssistantChatRoute/AssistantStudioLink';
import { JobProgressToolCall } from '@studio/routes/agents/AssistantChatRoute/JobProgressToolCall';

interface AssistantToolCallPartContentProps {
  readonly args: AssistantToolArgs;
  readonly argsText?: string;
  readonly isError?: boolean;
  readonly result?: unknown;
  readonly running?: boolean;
  readonly toolName: string;
}

type SerializedPart =
  | { readonly type: 'reasoning' | 'text'; readonly text: string }
  | {
      readonly type: 'tool-call';
      readonly args: AssistantToolArgs;
      readonly argsText?: string;
      readonly toolCallId: string;
      readonly toolName: string;
    };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

// Collapsed details and subtle groups store their parts and calls serialized in their args.
const getSerializedParts = (value: unknown): readonly SerializedPart[] =>
  (Array.isArray(value) ? value : []).filter(isRecord).flatMap((part): SerializedPart[] => {
    if ((part.type === 'text' || part.type === 'reasoning') && typeof part.text === 'string') {
      return part.text.trim() ? [{ type: part.type, text: part.text }] : [];
    }
    if (typeof part.toolName !== 'string') return [];
    return [
      {
        type: 'tool-call',
        args: isRecord(part.args) ? toAssistantToolArgs(part.args) : {},
        argsText: typeof part.argsText === 'string' ? part.argsText : undefined,
        toolCallId: typeof part.toolCallId === 'string' ? part.toolCallId : part.toolName,
        toolName: part.toolName,
      },
    ];
  });

const getPartKey = (part: SerializedPart) =>
  part.type === 'tool-call' ? part.toolCallId : part.type;

// Calls of the same tool in a subtle group fold into one step ("Read 3 files"), in the order the
// tool first appeared.
const groupRepeatedToolCalls = (parts: readonly SerializedPart[]): readonly SerializedPart[][] => {
  const groups: SerializedPart[][] = [];
  const byTool = new Map<string, SerializedPart[]>();
  for (const part of parts) {
    const existing = part.type === 'tool-call' ? byTool.get(part.toolName) : undefined;
    if (existing) {
      existing.push(part);
      continue;
    }
    const group = [part];
    if (part.type === 'tool-call') byTool.set(part.toolName, group);
    groups.push(group);
  }
  return groups;
};

const SerializedPartContent = ({ part }: { readonly part: SerializedPart }) => {
  if (part.type !== 'tool-call') {
    return part.type === 'reasoning' ? (
      <ThoughtStep text={part.text} />
    ) : (
      <div className="px-density-md py-density-xs">
        <MessageContent content={part.text} markdownLinkComponent={AssistantStudioLink} />
      </div>
    );
  }
  return (
    <AssistantToolCallPartContent
      args={part.args}
      argsText={part.argsText}
      toolName={part.toolName}
    />
  );
};

const AssistantToolCallPartContent = ({
  args,
  argsText,
  isError,
  result,
  running,
  toolName,
}: AssistantToolCallPartContentProps) => {
  if (
    toolName === ASSISTANT_THINKING_TOOL_NAME ||
    toolName === ASSISTANT_COLLAPSED_THINKING_TOOL_NAME
  ) {
    return typeof args.text === 'string' ? <ThoughtStep text={args.text} /> : null;
  }

  if (isAssistantJobProgressToolName(toolName)) return <JobProgressToolCall args={args} />;

  if (toolName === ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME) {
    return (
      <>
        {getSerializedParts(args.parts).map((part, index) => (
          <SerializedPartContent key={`${getPartKey(part)}-${index}`} part={part} />
        ))}
      </>
    );
  }

  if (toolName === ASSISTANT_SUBTLE_TOOL_GROUP_NAME) {
    return (
      <>
        {groupRepeatedToolCalls(getSerializedParts(args.actions)).map((group, index) => {
          const [first] = group;
          if (!first) return null;
          const key = `${getPartKey(first)}-${index}`;
          return group.length > 1 && first.type === 'tool-call' ? (
            <RepeatedToolCallsStep count={group.length} key={key} toolName={first.toolName}>
              {group.map((part, partIndex) => (
                <SerializedPartContent key={`${getPartKey(part)}-${partIndex}`} part={part} />
              ))}
            </RepeatedToolCallsStep>
          ) : (
            <SerializedPartContent key={key} part={first} />
          );
        })}
      </>
    );
  }

  return (
    <ToolCallStep
      args={args}
      argsText={argsText}
      isError={isError}
      result={result}
      running={running}
      toolName={toolName}
    />
  );
};

/**
 * Renders the NeMo Assistant's tool-call parts as rows in the chat's work group, unpacking the
 * runtime's synthetic thinking, subtle-group, and collapsed-details parts into their steps.
 */
export const AssistantToolCallPart: ToolCallMessagePartComponent<AssistantToolArgs, unknown> = ({
  args,
  argsText,
  isError,
  result,
  status,
  toolName,
}) => (
  <AssistantToolCallPartContent
    args={args}
    argsText={argsText}
    isError={isError}
    result={result}
    running={status.type === 'running'}
    toolName={toolName}
  />
);
