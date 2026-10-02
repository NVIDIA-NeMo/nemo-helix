// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ToolCallMessagePartComponent } from '@assistant-ui/react';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import { Text } from '@nvidia/foundations-react-core';
import { AssistantStudioLink } from '@studio/routes/agents/AssistantChatRoute/AssistantStudioLink';
import { JobProgressToolCall } from '@studio/routes/agents/AssistantChatRoute/JobProgressToolCall';
import { CollapsedThinkingToolCall } from '@studio/routes/agents/AssistantChatRoute/toolCall/CollapsedThinkingToolCall';
import { TOOL_LABELS } from '@studio/routes/agents/AssistantChatRoute/toolCall/constants';
import { FileChangeToolCallCard } from '@studio/routes/agents/AssistantChatRoute/toolCall/FileChangeToolCallCard';
import {
  formatSubtleToolMessage,
  getFileChangeSummary,
  getStringArg,
  getSubtleToolDetail,
  getSubtleToolGroupActions,
  getSubtleToolIcon,
  getSubtleToolMessage,
  getToolInvocation,
  getToolSummary,
} from '@studio/routes/agents/AssistantChatRoute/toolCall/helpers';
import { SubtleToolCallRow } from '@studio/routes/agents/AssistantChatRoute/toolCall/SubtleToolCallRow';
import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
  ASSISTANT_WORK_DETAILS_LABEL,
  isAssistantJobProgressToolName,
  toAssistantToolArgs,
  type AssistantToolArgs,
} from '@studio/routes/agents/AssistantChatRoute/toolParts';
import { ChevronRight, ClipboardList } from 'lucide-react';

interface AssistantToolCallPartContentProps {
  readonly args: AssistantToolArgs;
  readonly argsText: string;
  readonly toolName: string;
}

interface CollapsedStudioDetailsTextPart {
  readonly text: string;
  readonly type: 'text';
}

interface CollapsedStudioDetailsToolPart {
  readonly args: AssistantToolArgs;
  readonly argsText: string;
  readonly toolCallId: string;
  readonly toolName: string;
  readonly type: 'tool-call';
}

type CollapsedStudioDetailsPart = CollapsedStudioDetailsTextPart | CollapsedStudioDetailsToolPart;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const getCollapsedStudioDetailsParts = (
  args: AssistantToolArgs
): readonly CollapsedStudioDetailsPart[] => {
  const parts = args.parts;
  if (!Array.isArray(parts)) return [];

  return parts
    .map((part): CollapsedStudioDetailsPart | undefined => {
      if (!isRecord(part)) return undefined;

      if (part.type === 'text' && typeof part.text === 'string' && part.text.trim()) {
        return { type: 'text', text: part.text };
      }

      if (
        part.type === 'tool-call' &&
        typeof part.toolName === 'string' &&
        typeof part.toolCallId === 'string'
      ) {
        const toolArgs = isRecord(part.args) ? toAssistantToolArgs(part.args) : {};
        return {
          type: 'tool-call',
          args: toolArgs,
          argsText: typeof part.argsText === 'string' ? part.argsText : JSON.stringify(toolArgs),
          toolCallId: part.toolCallId,
          toolName: part.toolName,
        };
      }

      return undefined;
    })
    .filter((part): part is CollapsedStudioDetailsPart => part !== undefined);
};

const CollapsedStudioDetailsText = ({ text }: { readonly text: string }) => (
  <MessageContent content={text} markdownLinkComponent={AssistantStudioLink} />
);

const CollapsedStudioDetailsPartContent = ({
  part,
}: {
  readonly part: CollapsedStudioDetailsPart;
}) => {
  if (part.type === 'text') return <CollapsedStudioDetailsText text={part.text} />;

  return (
    <AssistantToolCallPartContent
      args={part.args}
      argsText={part.argsText}
      toolName={part.toolName}
    />
  );
};

const CollapsedStudioDetailsToolCall = ({ args }: { readonly args: AssistantToolArgs }) => {
  const storedLabel = getStringArg(args, ['label']);
  const label =
    !storedLabel || storedLabel.toLowerCase() === 'worked for unknown'
      ? ASSISTANT_WORK_DETAILS_LABEL
      : storedLabel;
  const parts = getCollapsedStudioDetailsParts(args);
  const contextLabel = label === ASSISTANT_WORK_DETAILS_LABEL ? undefined : label;
  if (!parts.length) return null;

  return (
    <Text asChild kind="body/regular/sm">
      <details
        className="group/studio-details my-density-xs max-w-full text-gray-500 dark:text-gray-400"
        data-testid="assistant-collapsed-studio-details"
      >
        <summary className="inline-flex cursor-pointer list-none items-center gap-density-xs marker:hidden">
          <ChevronRight
            aria-hidden
            className="size-3 shrink-0 transition-transform group-open/studio-details:rotate-90"
          />
          <ClipboardList aria-hidden className="size-3.5 shrink-0" />
          <span>
            <span className="font-medium text-primary">View work</span>
            {contextLabel ? ` · ${contextLabel}` : null}
          </span>
        </summary>
        <div
          className="mt-density-xs space-y-density-xs border-l border-base pl-density-md text-secondary"
          data-testid="assistant-collapsed-studio-details-content"
        >
          {parts.map((part, index) => (
            <CollapsedStudioDetailsPartContent
              key={`${part.type}-${part.type === 'text' ? part.text.slice(0, 24) : part.toolCallId}-${index}`}
              part={part}
            />
          ))}
        </div>
      </details>
    </Text>
  );
};

const AssistantToolCallPartContent = ({
  args,
  argsText,
  toolName,
}: AssistantToolCallPartContentProps) => {
  if (toolName === ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME) {
    return <CollapsedStudioDetailsToolCall args={args} />;
  }

  if (toolName === ASSISTANT_COLLAPSED_THINKING_TOOL_NAME) {
    const text = getStringArg(args, ['text']);
    return text ? <CollapsedThinkingToolCall text={text} /> : null;
  }

  if (toolName === ASSISTANT_THINKING_TOOL_NAME) {
    const text = getStringArg(args, ['text']);
    return text ? <CollapsedThinkingToolCall label="Thinking" text={text} /> : null;
  }

  if (isAssistantJobProgressToolName(toolName)) {
    return <JobProgressToolCall args={args} />;
  }

  if (toolName === ASSISTANT_SUBTLE_TOOL_GROUP_NAME) {
    const subtleActions = getSubtleToolGroupActions(args);
    return subtleActions.length ? <SubtleToolCallRow actions={subtleActions} /> : null;
  }

  const subtleMessage = getSubtleToolMessage(toolName, args);
  if (subtleMessage) {
    return (
      <SubtleToolCallRow
        actions={[
          {
            detail: getSubtleToolDetail(toolName, args, subtleMessage),
            Icon: getSubtleToolIcon(toolName),
            invocation: getToolInvocation(toolName, args),
            message: subtleMessage,
            toolCallId: toolName,
            toolName,
          },
        ]}
      />
    );
  }

  if (toolName === 'Write' || toolName === 'Edit' || toolName === 'MultiEdit') {
    const fileChangeSummary = getFileChangeSummary(toolName, args, argsText);
    if (fileChangeSummary) {
      return <FileChangeToolCallCard summary={fileChangeSummary} />;
    }
  }

  const label = TOOL_LABELS[toolName] ?? toolName;
  const fallbackMessage = formatSubtleToolMessage(
    `Used ${label}`,
    getToolSummary(toolName, args),
    `Used ${label}`
  );
  return (
    <SubtleToolCallRow
      actions={[
        {
          detail: getSubtleToolDetail(toolName, args, fallbackMessage),
          Icon: getSubtleToolIcon(toolName),
          invocation: getToolInvocation(toolName, args),
          message: fallbackMessage,
          toolCallId: toolName,
          toolName,
        },
      ]}
    />
  );
};

export const AssistantToolCallPart: ToolCallMessagePartComponent<AssistantToolArgs, unknown> = ({
  args,
  argsText,
  toolName,
}) => <AssistantToolCallPartContent args={args} argsText={argsText} toolName={toolName} />;
