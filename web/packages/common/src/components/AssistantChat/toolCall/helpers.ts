// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  SUBTLE_MESSAGE_MAX_LENGTH,
  SUBTLE_TOOL_ICONS,
  TOOL_ICONS,
  TOOL_LABELS,
} from '@nemo/common/src/components/AssistantChat/toolCall/constants';
import type {
  FileChangeSummary,
  SubtleToolAction,
} from '@nemo/common/src/components/AssistantChat/toolCall/types';
import {
  isAssistantSubtleToolCallName,
  toAssistantToolArgs,
  type AssistantToolArgs,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import type { NvidiaGuiIconProps } from '@nemo/icons';
import { WindowTerminal } from '@nemo/icons/icons';
import type { FC } from 'react';

const getStringArg = (args: AssistantToolArgs, keys: string[]): string | undefined => {
  for (const key of keys) {
    const value = args[key];
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  return undefined;
};

const getRawStringArg = (args: Record<string, unknown>, keys: string[]): string | undefined => {
  for (const key of keys) {
    const value = args[key];
    if (typeof value === 'string') return value;
  }
  return undefined;
};

export const getToolSummary = (toolName: string, args: AssistantToolArgs): string | undefined => {
  switch (toolName) {
    case 'Bash':
      return getStringArg(args, ['command']);
    case 'Edit':
    case 'MultiEdit':
    case 'Read':
    case 'Write':
      return getStringArg(args, ['file_path', 'path']);
    case 'Glob':
      return getStringArg(args, ['pattern']);
    case 'Grep': {
      const pattern = getStringArg(args, ['pattern']);
      const path = getStringArg(args, ['path']);
      return [pattern, path].filter(Boolean).join(' in ') || undefined;
    }
    case 'LS':
      return getStringArg(args, ['path']);
    case 'TodoWrite': {
      const todos = args.todos;
      return Array.isArray(todos) ? `${todos.length} todos` : undefined;
    }
    case 'WebFetch':
      return getStringArg(args, ['url']);
    case 'WebSearch':
      return getStringArg(args, ['query']);
    default:
      return getStringArg(args, ['command', 'file_path', 'path', 'pattern', 'query', 'url']);
  }
};

export const compactSubtleDetail = (detail: string | undefined): string | undefined => {
  const compacted = detail?.replace(/\s+/g, ' ').trim();
  if (!compacted) return undefined;
  if (compacted.length <= SUBTLE_MESSAGE_MAX_LENGTH) return compacted;
  return `${compacted.slice(0, SUBTLE_MESSAGE_MAX_LENGTH - 3).trimEnd()}...`;
};

export const formatSubtleToolMessage = (
  action: string,
  detail: string | undefined,
  fallback: string
): string => {
  const compactedDetail = compactSubtleDetail(detail);
  return compactedDetail ? `${action} ${compactedDetail}` : fallback;
};

export const getSubtleToolIcon = (toolName: string): FC<Omit<NvidiaGuiIconProps, 'iconName'>> =>
  SUBTLE_TOOL_ICONS[toolName] ?? TOOL_ICONS[toolName] ?? WindowTerminal;

const getRepeatedSubtleToolMessage = (toolName: string, count: number): string => {
  switch (toolName) {
    case 'AskUserQuestion':
      return `Asked ${count} questions`;
    case 'Bash':
      return `Ran ${count} commands`;
    case 'FindFiles':
      return `Searched files ${count} times`;
    case 'Glob':
      return `Found files ${count} times`;
    case 'Grep':
      return `Searched text ${count} times`;
    case 'LS':
      return `Listed ${count} directories`;
    case 'Read':
      return `Read ${count} files`;
    case 'TaskCreate':
      return `Created ${count} tasks`;
    case 'TaskUpdate':
      return `Updated ${count} tasks`;
    case 'TodoWrite':
      return `Updated todos ${count} times`;
    case 'ToolSearch':
      return `Searched tools ${count} times`;
    case 'WebFetch':
      return `Fetched ${count} URLs`;
    case 'WebSearch':
      return `Searched web ${count} times`;
    default: {
      const label = TOOL_LABELS[toolName] ?? toolName;
      return `Used ${label} ${count} times`;
    }
  }
};

const getFileName = (path: string): string => {
  const segments = path.split(/[\\/]/).filter(Boolean);
  return segments.at(-1) ?? path;
};

const getLineCount = (content: string): number => {
  if (!content) return 0;

  const normalized = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const withoutTrailingNewline = normalized.endsWith('\n') ? normalized.slice(0, -1) : normalized;

  return withoutTrailingNewline.split('\n').length;
};

const getEditStats = (args: Record<string, unknown>): { additions: number; deletions: number } => ({
  additions: getLineCount(getRawStringArg(args, ['new_string']) ?? ''),
  deletions: getLineCount(getRawStringArg(args, ['old_string']) ?? ''),
});

const isToolArgsRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const splitCollapsedThinkingParagraphs = (text: string): readonly string[] =>
  text
    .trim()
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

const getAskUserQuestionSummary = (args: AssistantToolArgs): string | undefined => {
  const questions = args.questions;
  const firstQuestion = Array.isArray(questions) ? questions.find(isToolArgsRecord) : undefined;
  return (
    getStringArg(args, ['question', 'prompt']) ??
    (firstQuestion
      ? getRawStringArg(firstQuestion, ['question', 'prompt', 'header'])?.trim()
      : undefined)
  );
};

const formatArgs = (args: AssistantToolArgs, argsText: string): string => {
  const trimmedArgsText = argsText.trim();
  if (trimmedArgsText && trimmedArgsText !== '{}') return trimmedArgsText;
  return JSON.stringify(args, null, 2);
};

export const getFileChangeSummary = (
  toolName: string,
  args: AssistantToolArgs,
  argsText: string
): FileChangeSummary | undefined => {
  const path = getStringArg(args, ['file_path', 'path']);
  if (!path) return undefined;

  if (toolName === 'Write') {
    const content = getRawStringArg(args, ['content']);
    if (content === undefined) return undefined;

    return {
      action: 'Wrote',
      additions: getLineCount(content),
      deletions: 0,
      path,
      reviewContent: content,
    };
  }

  if (toolName === 'Edit') {
    return {
      action: 'Edited',
      ...getEditStats(args),
      path,
      reviewContent: formatArgs(args, argsText),
    };
  }

  if (toolName === 'MultiEdit') {
    const edits = args.edits;
    if (!Array.isArray(edits)) return undefined;

    const stats = edits.filter(isToolArgsRecord).reduce<{ additions: number; deletions: number }>(
      (total, edit) => {
        const editStats = getEditStats(edit);
        return {
          additions: total.additions + editStats.additions,
          deletions: total.deletions + editStats.deletions,
        };
      },
      { additions: 0, deletions: 0 }
    );

    return {
      action: 'Edited',
      ...stats,
      path,
      reviewContent: formatArgs(args, argsText),
    };
  }

  return undefined;
};

export const getSubtleToolMessage = (
  toolName: string,
  args: AssistantToolArgs
): string | undefined => {
  if (!isAssistantSubtleToolCallName(toolName)) return undefined;

  if (toolName === 'AskUserQuestion') {
    return formatSubtleToolMessage('Asked', getAskUserQuestionSummary(args), 'Asked user question');
  }

  if (toolName === 'Bash') {
    return formatSubtleToolMessage(
      'Ran',
      getStringArg(args, ['description', 'command']),
      'Ran command'
    );
  }

  if (toolName === 'FindFiles') {
    return formatSubtleToolMessage(
      'Searched files',
      getStringArg(args, ['query', 'pattern', 'path']),
      'Searched files'
    );
  }

  if (toolName === 'Grep') {
    return formatSubtleToolMessage(
      'Searched text',
      getToolSummary(toolName, args),
      'Searched text'
    );
  }

  if (toolName === 'Glob') {
    return formatSubtleToolMessage('Found files', getToolSummary(toolName, args), 'Found files');
  }

  if (toolName === 'LS') {
    return formatSubtleToolMessage(
      'Listed directory',
      getToolSummary(toolName, args),
      'Listed directory'
    );
  }

  if (toolName === 'Read') {
    const path = getStringArg(args, ['file_path', 'path']);
    return path ? `Read ${getFileName(path)}` : 'Read file';
  }

  if (toolName === 'TaskCreate') {
    return formatSubtleToolMessage(
      'Created task',
      getStringArg(args, ['description', 'task', 'prompt', 'query']),
      'Created task'
    );
  }

  if (toolName === 'TaskUpdate') {
    return formatSubtleToolMessage(
      'Updated task',
      getStringArg(args, ['description', 'task', 'status']),
      'Updated task'
    );
  }

  if (toolName === 'TodoWrite') {
    return formatSubtleToolMessage(
      'Updated todos',
      getToolSummary(toolName, args),
      'Updated todos'
    );
  }

  if (toolName === 'ToolSearch') {
    return formatSubtleToolMessage(
      'Searched tools',
      getStringArg(args, ['query', 'pattern', 'name']),
      'Searched tools'
    );
  }

  if (toolName === 'WebFetch') {
    return formatSubtleToolMessage('Fetched URL', getToolSummary(toolName, args), 'Fetched URL');
  }

  if (toolName === 'WebSearch') {
    return formatSubtleToolMessage('Searched web', getToolSummary(toolName, args), 'Searched web');
  }

  const label = TOOL_LABELS[toolName] ?? toolName;
  return formatSubtleToolMessage(`Used ${label}`, getToolSummary(toolName, args), `Used ${label}`);
};

export const getSubtleToolDetail = (
  toolName: string,
  args: AssistantToolArgs,
  message: string
): string => {
  if (toolName === 'AskUserQuestion') {
    return compactSubtleDetail(getAskUserQuestionSummary(args)) ?? message;
  }

  if (toolName === 'Bash') {
    return compactSubtleDetail(getStringArg(args, ['description', 'command'])) ?? message;
  }

  if (toolName === 'FindFiles') {
    return compactSubtleDetail(getStringArg(args, ['query', 'pattern', 'path'])) ?? message;
  }

  if (toolName === 'Read') {
    const path = getStringArg(args, ['file_path', 'path']);
    return path ? getFileName(path) : message;
  }

  if (toolName === 'TaskCreate') {
    return (
      compactSubtleDetail(getStringArg(args, ['description', 'task', 'prompt', 'query'])) ?? message
    );
  }

  if (toolName === 'TaskUpdate') {
    return compactSubtleDetail(getStringArg(args, ['description', 'task', 'status'])) ?? message;
  }

  if (toolName === 'ToolSearch') {
    return compactSubtleDetail(getStringArg(args, ['query', 'pattern', 'name'])) ?? message;
  }

  return compactSubtleDetail(getToolSummary(toolName, args)) ?? message;
};

export const getToolInvocation = (toolName: string, args: AssistantToolArgs): string => {
  if (toolName === 'Bash') {
    const command = args.command;
    if (typeof command === 'string' && command) return command;
  }

  return JSON.stringify(args, null, 2);
};

export const getSubtleToolGroupActions = (args: AssistantToolArgs): readonly SubtleToolAction[] => {
  const actions = args.actions;
  if (!Array.isArray(actions)) return [];

  return actions
    .filter(isToolArgsRecord)
    .map((action, index): SubtleToolAction | undefined => {
      const toolName = getRawStringArg(action, ['toolName'])?.trim();
      if (!toolName) return undefined;

      const actionArgs = isToolArgsRecord(action.args) ? toAssistantToolArgs(action.args) : {};
      const message = getSubtleToolMessage(toolName, actionArgs);
      if (!message) return undefined;

      return {
        detail: getSubtleToolDetail(toolName, actionArgs, message),
        Icon: getSubtleToolIcon(toolName),
        invocation: getToolInvocation(toolName, actionArgs),
        message,
        toolCallId: getRawStringArg(action, ['toolCallId'])?.trim() ?? `${toolName}-${index}`,
        toolName,
      };
    })
    .filter((action): action is SubtleToolAction => action !== undefined);
};

export const summarizeRepeatedSubtleToolActions = (
  actions: readonly SubtleToolAction[]
): readonly SubtleToolAction[] => {
  const groupedActions = new Map<string, SubtleToolAction[]>();

  for (const action of actions) {
    const existingActions = groupedActions.get(action.toolName);
    if (existingActions) {
      existingActions.push(action);
    } else {
      groupedActions.set(action.toolName, [action]);
    }
  }

  return Array.from(groupedActions.values()).map((group) => {
    if (group.length === 1) return group[0]!;

    const firstAction = group[0]!;
    return {
      ...firstAction,
      details: group.map((action) => action.detail),
      invocations: group.map((action) => action.invocation),
      message: getRepeatedSubtleToolMessage(firstAction.toolName, group.length),
      title: group.map((action) => action.message).join(' | '),
      toolCallId: `${firstAction.toolCallId}-${group.length}`,
    };
  });
};

export { getStringArg };
