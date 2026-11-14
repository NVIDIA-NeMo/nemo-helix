// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ToolCallMessagePartComponent } from '@assistant-ui/react';
import {
  getSubtleToolGroupActions,
  groupRepeatedSubtleToolActions,
} from '@nemo/common/src/components/AssistantChat/toolCall/helpers';
import type { SubtleToolAction } from '@nemo/common/src/components/AssistantChat/toolCall/types';
import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
  isAssistantJobProgressToolName,
  toAssistantToolArgs,
  type AssistantToolArgs,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import { RepeatedToolCallsStep } from '@nemo/common/src/components/AssistantChat/work/RepeatedToolCallsStep';
import { ThoughtStep } from '@nemo/common/src/components/AssistantChat/work/ThoughtStep';
import { ToolCallStep } from '@nemo/common/src/components/AssistantChat/work/ToolCallStep';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import { isPlainObject } from '@nemo/common/src/utils/isPlainObject';
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

type CollapsedDetailsPart =
  | { readonly type: 'reasoning' | 'text'; readonly text: string }
  | {
      readonly type: 'tool-call';
      readonly args: AssistantToolArgs;
      readonly argsText?: string;
      readonly isError?: boolean;
      readonly result?: unknown;
      readonly toolCallId: string;
      readonly toolName: string;
    };

const getCollapsedDetailsParts = (value: unknown): readonly CollapsedDetailsPart[] =>
  (Array.isArray(value) ? value : [])
    .filter(isPlainObject)
    .flatMap((part): CollapsedDetailsPart[] => {
      if ((part.type === 'text' || part.type === 'reasoning') && typeof part.text === 'string') {
        if (!part.text.trim()) {
          return [];
        }
        return [{ type: part.type, text: part.text }];
      }
      if (part.type !== 'tool-call' || typeof part.toolName !== 'string') {
        return [];
      }
      return [
        {
          type: 'tool-call',
          args: isPlainObject(part.args) ? toAssistantToolArgs(part.args) : {},
          argsText: typeof part.argsText === 'string' ? part.argsText : undefined,
          isError: part.isError === true,
          result: part.result,
          toolCallId: typeof part.toolCallId === 'string' ? part.toolCallId : part.toolName,
          toolName: part.toolName,
        },
      ];
    });

const getCollapsedDetailsPartKey = (part: CollapsedDetailsPart, index: number) =>
  `${part.type === 'tool-call' ? part.toolCallId : part.type}-${index}`;

const CollapsedDetailsPartContent = ({ part }: { readonly part: CollapsedDetailsPart }) => {
  switch (part.type) {
    case 'reasoning':
      return <ThoughtStep text={part.text} />;
    case 'text':
      return (
        <div className="py-density-xs">
          <MessageContent content={part.text} markdownLinkComponent={AssistantStudioLink} />
        </div>
      );
    default:
      return (
        <AssistantToolCallPartContent
          args={part.args}
          argsText={part.argsText}
          isError={part.isError}
          result={part.result}
          toolName={part.toolName}
        />
      );
  }
};

interface SubtleToolGroupProps {
  readonly args: AssistantToolArgs;
  readonly running?: boolean;
}

const SubtleToolGroup = ({ args, running }: SubtleToolGroupProps) => {
  const actions = getSubtleToolGroupActions(args);
  const runningAction = running ? actions.at(-1) : undefined;
  const renderStep = (action: SubtleToolAction) => (
    <ToolCallStep
      key={action.toolCallId}
      args={action.args}
      running={action === runningAction}
      toolName={action.toolName}
    />
  );

  return (
    <>
      {groupRepeatedSubtleToolActions(actions).map((group) => {
        const [first] = group.actions;
        if (!group.repeatedTitle) {
          return renderStep(first);
        }
        return (
          <RepeatedToolCallsStep
            key={first.toolCallId}
            defaultOpen={runningAction !== undefined && group.actions.includes(runningAction)}
            title={group.repeatedTitle}
            toolName={first.toolName}
          >
            {group.actions.map(renderStep)}
          </RepeatedToolCallsStep>
        );
      })}
    </>
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
    if (typeof args.text !== 'string') {
      return null;
    }
    return <ThoughtStep text={args.text} />;
  }

  if (isAssistantJobProgressToolName(toolName)) {
    return <JobProgressToolCall args={args} />;
  }

  if (toolName === ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME) {
    return (
      <>
        {getCollapsedDetailsParts(args.parts).map((part, index) => (
          <CollapsedDetailsPartContent key={getCollapsedDetailsPartKey(part, index)} part={part} />
        ))}
      </>
    );
  }

  if (toolName === ASSISTANT_SUBTLE_TOOL_GROUP_NAME) {
    return <SubtleToolGroup args={args} running={running} />;
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

/** Renders NeMo Assistant tool calls as work steps, unpacking the runtime's synthetic parts. */
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
