// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadAssistantMessagePart } from '@assistant-ui/react';
import {
  ASSISTANT_WORK_DETAILS_LABEL,
  createAssistantCollapsedStudioDetailsPart,
  createAssistantToolCallPart,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import type { ChatCompletionChunk } from 'openai/resources/index.mjs';

type ToolCallDelta = ChatCompletionChunk.Choice.Delta.ToolCall;
type ToolCallPart = Extract<ThreadAssistantMessagePart, { type: 'tool-call' }>;

export interface CompletionDeltaParts {
  readonly reasoning?: string;
  readonly text?: string;
  readonly toolCalls?: readonly ToolCallDelta[];
}

const isToolCallPart = (part: ThreadAssistantMessagePart): part is ToolCallPart =>
  part.type === 'tool-call';

const parseToolArgs = (argsText: string): unknown => {
  try {
    return JSON.parse(argsText) as unknown;
  } catch {
    // Arguments stream as partial JSON; keep them empty until the payload parses.
    return {};
  }
};

const appendReasoning = (
  parts: ThreadAssistantMessagePart[],
  reasoning: string
): ThreadAssistantMessagePart[] => {
  const last = parts.at(-1);
  if (last?.type === 'reasoning') {
    parts[parts.length - 1] = { ...last, text: last.text + reasoning };
    return parts;
  }
  return [...parts, { type: 'reasoning', text: reasoning }];
};

const appendText = (
  parts: ThreadAssistantMessagePart[],
  text: string
): ThreadAssistantMessagePart[] => {
  const last = parts.at(-1);
  if (last?.type === 'text') {
    parts[parts.length - 1] = { ...last, text: last.text + text };
    return parts;
  }
  return [...parts, { type: 'text', text }];
};

// OpenAI streams each tool call as fragments keyed by `index`; only the first fragment carries
// the id and name, so later fragments are matched to the index-th tool-call part.
const appendToolCall = (
  parts: ThreadAssistantMessagePart[],
  { index, id, function: fn }: ToolCallDelta
): ThreadAssistantMessagePart[] => {
  const existing = parts.filter(isToolCallPart)[index];
  if (!existing) {
    const argsText = fn?.arguments ?? '';
    return [
      ...parts,
      createAssistantToolCallPart({
        argsText,
        input: parseToolArgs(argsText),
        toolCallId: id || `tool-call-${index}`,
        toolName: fn?.name || 'tool',
      }),
    ];
  }

  const argsText = existing.argsText + (fn?.arguments ?? '');
  parts[parts.indexOf(existing)] = createAssistantToolCallPart({
    argsText,
    input: parseToolArgs(argsText),
    toolCallId: existing.toolCallId,
    toolName: existing.toolName,
  });
  return parts;
};

/** Folds one chat-completion delta into the assistant message parts, in stream order. */
export const appendCompletionDeltaParts = (
  parts: readonly ThreadAssistantMessagePart[],
  { reasoning, text, toolCalls = [] }: CompletionDeltaParts
): readonly ThreadAssistantMessagePart[] => {
  let next = [...parts];
  if (reasoning) next = appendReasoning(next, reasoning);
  if (text) next = appendText(next, text);
  for (const toolCall of toolCalls) next = appendToolCall(next, toolCall);
  return next;
};

const isReasoningPart = (part: ThreadAssistantMessagePart): boolean => part.type === 'reasoning';

/**
 * Collapses reasoning and tool calls into a single "Work details" group, leaving the text after
 * the last tool call as the visible answer. Replies without tool calls are returned unchanged.
 */
export const getWorkDetailsMessageParts = (
  parts: readonly ThreadAssistantMessagePart[]
): readonly ThreadAssistantMessagePart[] => {
  const lastToolIndex = parts.findLastIndex(isToolCallPart);
  if (lastToolIndex < 0) return parts;

  const trailing = parts.slice(lastToolIndex + 1);
  const details = createAssistantCollapsedStudioDetailsPart({
    label: ASSISTANT_WORK_DETAILS_LABEL,
    parts: [...parts.slice(0, lastToolIndex + 1), ...trailing.filter(isReasoningPart)],
  });
  const final = trailing.filter((part) => !isReasoningPart(part));
  return details ? [details, ...final] : final;
};
