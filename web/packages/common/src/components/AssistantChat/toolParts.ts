// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadAssistantMessagePart, ThreadMessageLike } from '@assistant-ui/react';

type AssistantToolCallPart = Extract<ThreadAssistantMessagePart, { type: 'tool-call' }>;
export type AssistantToolArgs = AssistantToolCallPart['args'];
type AssistantToolArgValue = AssistantToolArgs[string];

export const ASSISTANT_COLLAPSED_THINKING_TOOL_NAME = 'AssistantCollapsedThinking';
export const ASSISTANT_THINKING_TOOL_NAME = 'AssistantThinking';
export const ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME = 'AssistantCollapsedStudioDetails';
export const ASSISTANT_SUBTLE_TOOL_GROUP_NAME = 'AssistantSubtleToolGroup';
export const ASSISTANT_WORK_DETAILS_LABEL = 'Work details';

export const ASSISTANT_JOB_PROGRESS_TOOL_NAME = 'job_progress';
export const ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME = 'mcp__nemo_studio__job_progress';

const FILE_CHANGE_TOOL_CALL_NAMES = new Set(['Edit', 'MultiEdit', 'Write']);

export const isAssistantJobProgressToolName = (toolName: string): boolean =>
  toolName === ASSISTANT_JOB_PROGRESS_TOOL_NAME ||
  toolName === ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME;

export const isAssistantSubtleToolCallName = (toolName: string): boolean =>
  toolName !== ASSISTANT_COLLAPSED_THINKING_TOOL_NAME &&
  toolName !== ASSISTANT_THINKING_TOOL_NAME &&
  toolName !== ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME &&
  toolName !== ASSISTANT_SUBTLE_TOOL_GROUP_NAME &&
  !FILE_CHANGE_TOOL_CALL_NAMES.has(toolName) &&
  !isAssistantJobProgressToolName(toolName);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const toAssistantToolArgValue = (value: unknown): AssistantToolArgValue | undefined => {
  if (value === null) return value;
  if (typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? value : undefined;

  if (Array.isArray(value)) {
    return value
      .map(toAssistantToolArgValue)
      .filter((item): item is AssistantToolArgValue => item !== undefined);
  }

  if (isRecord(value)) return toAssistantToolArgs(value);

  return undefined;
};

export const toAssistantToolArgs = (input: unknown): AssistantToolArgs => {
  if (!isRecord(input)) return {};

  const args: Record<string, AssistantToolArgValue> = {};
  for (const [key, value] of Object.entries(input)) {
    const nextValue = toAssistantToolArgValue(value);
    if (nextValue !== undefined) args[key] = nextValue;
  }

  return args;
};

export const createAssistantToolCallPart = ({
  argsText,
  input,
  toolCallId,
  toolName,
}: {
  argsText?: string;
  input: unknown;
  toolCallId: string;
  toolName: string;
}): ThreadAssistantMessagePart => {
  const args = toAssistantToolArgs(input);
  return {
    type: 'tool-call',
    toolCallId,
    toolName,
    args,
    argsText: argsText ?? JSON.stringify(args),
  };
};

export const createAssistantThinkingPart = (
  text: string,
  toolCallId: string
): ThreadAssistantMessagePart => {
  const args = toAssistantToolArgs({ text });

  return {
    type: 'tool-call',
    toolCallId,
    toolName: ASSISTANT_THINKING_TOOL_NAME,
    args,
    argsText: JSON.stringify(args),
  };
};

type AssistantThinkingPart = AssistantToolCallPart & {
  toolName: typeof ASSISTANT_THINKING_TOOL_NAME;
};

export const isAssistantThinkingPart = (
  part: ThreadAssistantMessagePart
): part is AssistantThinkingPart =>
  part.type === 'tool-call' && part.toolName === ASSISTANT_THINKING_TOOL_NAME;

export const getAssistantThinkingPartText = (part: ThreadAssistantMessagePart): string => {
  if (!isAssistantThinkingPart(part)) return '';
  const text = part.args.text;
  return typeof text === 'string' ? text : '';
};

export const createAssistantCollapsedThinkingPart = (text: string): ThreadAssistantMessagePart => {
  const args = toAssistantToolArgs({ text });

  return {
    type: 'tool-call',
    toolCallId: 'assistant-collapsed-thinking',
    toolName: ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
    args,
    argsText: JSON.stringify(args),
  };
};

interface SerializedAssistantTextPart {
  readonly text: string;
  readonly type: 'text';
}

interface SerializedAssistantToolPart {
  readonly args: AssistantToolArgs;
  readonly argsText: string;
  readonly toolCallId: string;
  readonly toolName: string;
  readonly type: 'tool-call';
}

interface SerializedAssistantReasoningPart {
  readonly text: string;
  readonly type: 'reasoning';
}

type SerializedAssistantPart =
  | SerializedAssistantReasoningPart
  | SerializedAssistantTextPart
  | SerializedAssistantToolPart;

const serializeAssistantPart = (
  part: ThreadAssistantMessagePart
): SerializedAssistantPart | undefined => {
  if (part.type === 'text' || part.type === 'reasoning') {
    const text = part.text.trim();
    return text ? { type: part.type, text } : undefined;
  }

  if (part.type !== 'tool-call' || part.toolName === ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME) {
    return undefined;
  }

  return {
    type: 'tool-call',
    args: toAssistantToolArgs(part.args),
    argsText: part.argsText,
    toolCallId: part.toolCallId,
    toolName: part.toolName,
  };
};

export const createAssistantCollapsedStudioDetailsPart = ({
  label,
  parts,
}: {
  readonly label: string;
  readonly parts: readonly ThreadAssistantMessagePart[];
}): ThreadAssistantMessagePart | undefined => {
  const serializedParts = groupConsecutiveAssistantSubtleToolCalls(parts)
    .map(serializeAssistantPart)
    .filter((part): part is SerializedAssistantPart => part !== undefined);
  if (!serializedParts.length) return undefined;

  const args = toAssistantToolArgs({ label, parts: serializedParts });
  return {
    type: 'tool-call',
    toolCallId: 'assistant-collapsed-studio-details',
    toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
    args,
    argsText: JSON.stringify(args),
  };
};

interface AssistantSubtleToolAction {
  readonly args: AssistantToolArgs;
  readonly toolCallId: string;
  readonly toolName: string;
}

interface AssistantSubtleToolPartGroup {
  readonly actions: readonly AssistantSubtleToolAction[];
  readonly part: ThreadAssistantMessagePart;
}

const getGroupedSubtleToolActions = (
  part: AssistantToolCallPart
): readonly AssistantSubtleToolAction[] => {
  const actions = part.args.actions;
  if (!Array.isArray(actions)) return [];

  return actions
    .filter(isRecord)
    .map((action): AssistantSubtleToolAction | undefined => {
      const toolCallId = typeof action.toolCallId === 'string' ? action.toolCallId : undefined;
      const toolName = typeof action.toolName === 'string' ? action.toolName : undefined;
      if (!toolCallId || !toolName || !isAssistantSubtleToolCallName(toolName)) return undefined;

      return {
        args: isRecord(action.args) ? toAssistantToolArgs(action.args) : {},
        toolCallId,
        toolName,
      };
    })
    .filter((action): action is AssistantSubtleToolAction => action !== undefined);
};

const getSubtleToolActions = (
  part: ThreadAssistantMessagePart
): readonly AssistantSubtleToolAction[] | undefined => {
  if (part.type !== 'tool-call') return undefined;

  if (part.toolName === ASSISTANT_SUBTLE_TOOL_GROUP_NAME) {
    const actions = getGroupedSubtleToolActions(part);
    return actions.length ? actions : undefined;
  }

  if (!isAssistantSubtleToolCallName(part.toolName)) return undefined;

  return [
    {
      args: part.args,
      toolCallId: part.toolCallId,
      toolName: part.toolName,
    },
  ];
};

const isEmptyAssistantTextPart = (part: ThreadAssistantMessagePart): boolean =>
  part.type === 'text' && part.text.trim().length === 0;

export const isAssistantSubtleToolCallPart = (part: ThreadAssistantMessagePart): boolean =>
  getSubtleToolActions(part) !== undefined;

const createAssistantSubtleToolGroupPart = (
  actions: readonly AssistantSubtleToolAction[]
): ThreadAssistantMessagePart => {
  const firstAction = actions[0]!;
  const lastAction = actions[actions.length - 1]!;
  const args = toAssistantToolArgs({
    actions: actions.map((action) => ({
      args: action.args,
      toolCallId: action.toolCallId,
      toolName: action.toolName,
    })),
  });

  return {
    type: 'tool-call',
    toolCallId: `assistant-subtle-tools-${firstAction.toolCallId}-${lastAction.toolCallId}`,
    toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
    args,
    argsText: JSON.stringify(args),
  };
};

export const groupConsecutiveAssistantSubtleToolCalls = (
  parts: readonly ThreadAssistantMessagePart[]
): readonly ThreadAssistantMessagePart[] => {
  const groupedParts: ThreadAssistantMessagePart[] = [];
  let currentGroup: AssistantSubtleToolPartGroup[] = [];

  const flushGroup = () => {
    if (!currentGroup.length) return;

    const actions = currentGroup.flatMap((group) => group.actions);
    if (
      actions.length === 1 &&
      currentGroup.length === 1 &&
      currentGroup[0]!.part.type === 'tool-call' &&
      currentGroup[0]!.part.toolName !== ASSISTANT_SUBTLE_TOOL_GROUP_NAME
    ) {
      groupedParts.push(currentGroup[0]!.part);
    } else {
      groupedParts.push(createAssistantSubtleToolGroupPart(actions));
    }

    currentGroup = [];
  };

  for (const part of parts) {
    if (isEmptyAssistantTextPart(part)) continue;

    const actions = getSubtleToolActions(part);
    if (actions) {
      currentGroup.push({ actions, part });
      continue;
    }

    flushGroup();
    groupedParts.push(part);
  }

  flushGroup();
  return groupedParts;
};

const getToolOnlyAssistantMessageParts = (
  message: ThreadMessageLike
): readonly ThreadAssistantMessagePart[] | undefined => {
  if (message.role !== 'assistant' || !Array.isArray(message.content)) return undefined;

  const visibleParts = message.content.filter((part) => !isEmptyAssistantTextPart(part));
  if (!visibleParts.length || !visibleParts.every(isAssistantSubtleToolCallPart)) return undefined;
  return visibleParts;
};

export const mergeConsecutiveAssistantSubtleToolMessages = (
  messages: readonly ThreadMessageLike[]
): readonly ThreadMessageLike[] => {
  const mergedMessages: ThreadMessageLike[] = [];
  let pendingMessage: ThreadMessageLike | undefined;
  let pendingParts: ThreadAssistantMessagePart[] = [];

  const flushPendingMessage = () => {
    if (!pendingMessage) return;

    mergedMessages.push({
      ...pendingMessage,
      content: groupConsecutiveAssistantSubtleToolCalls(pendingParts),
    });
    pendingMessage = undefined;
    pendingParts = [];
  };

  for (const message of messages) {
    const toolOnlyParts = getToolOnlyAssistantMessageParts(message);
    if (!toolOnlyParts) {
      flushPendingMessage();
      mergedMessages.push(message);
      continue;
    }

    if (!pendingMessage) {
      pendingMessage = message;
    } else {
      pendingMessage = {
        ...pendingMessage,
        status: message.status ?? pendingMessage.status,
      };
    }
    pendingParts.push(...toolOnlyParts);
  }

  flushPendingMessage();
  return mergedMessages;
};
