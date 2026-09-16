// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadAssistantMessagePart, ThreadMessageLike } from '@assistant-ui/react';
import {
  ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME,
  ASSISTANT_JOB_PROGRESS_TOOL_NAME,
} from '@studio/routes/agents/AssistantChatRoute/jobProgressConsts';

type AssistantToolCallPart = Extract<ThreadAssistantMessagePart, { type: 'tool-call' }>;
export type AssistantToolArgs = AssistantToolCallPart['args'];
type AssistantToolArgValue = AssistantToolArgs[string];

export const ASSISTANT_COLLAPSED_THINKING_TOOL_NAME = 'AssistantCollapsedThinking';
export const ASSISTANT_THINKING_TOOL_NAME = 'AssistantThinking';
export const ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME = 'AssistantCollapsedStudioDetails';
export const ASSISTANT_SUBTLE_TOOL_GROUP_NAME = 'AssistantSubtleToolGroup';
export const ASSISTANT_WORK_DETAILS_LABEL = 'Work details';
export const STUDIO_MESSAGE_SUMMARY_START = '<<<NEMO_STUDIO_MESSAGE_SUMMARY_V1>>>';
export const STUDIO_MESSAGE_SUMMARY_END = '<<<END_NEMO_STUDIO_MESSAGE_SUMMARY_V1>>>';

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
  input,
  toolCallId,
  toolName,
}: {
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
    argsText: JSON.stringify(args),
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

const createAssistantCollapsedThinkingPart = (text: string): ThreadAssistantMessagePart => {
  const args = toAssistantToolArgs({ text });

  return {
    type: 'tool-call',
    toolCallId: 'assistant-collapsed-thinking',
    toolName: ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
    args,
    argsText: JSON.stringify(args),
  };
};

interface AssistantCompletedMessageOptions {
  readonly elapsedMs?: number;
}

interface StudioSummaryBlock {
  readonly detailsLabel: string;
  readonly detailParts: readonly ThreadAssistantMessagePart[];
  readonly finalParts: readonly ThreadAssistantMessagePart[];
  readonly summaryText: string;
}

interface MarkdownLink {
  readonly href: string;
  readonly markdown: string;
}

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

type SerializedAssistantPart = SerializedAssistantTextPart | SerializedAssistantToolPart;

const formatElapsedDuration = (elapsedMs: number | undefined): string | undefined => {
  if (elapsedMs === undefined || !Number.isFinite(elapsedMs)) return undefined;

  const totalSeconds = Math.max(1, Math.round(elapsedMs / 1000));
  if (totalSeconds < 60) return `${totalSeconds}s`;

  const totalMinutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (totalMinutes < 60) return seconds ? `${totalMinutes}m ${seconds}s` : `${totalMinutes}m`;

  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return minutes ? `${hours}h ${minutes}m` : `${hours}h`;
};

const normalizeSummaryField = (value: string | undefined): string | undefined => {
  const trimmed = value?.replace(/\s+/g, ' ').trim();
  return trimmed || undefined;
};

const normalizeMarkdownSummary = (value: string | undefined): string | undefined => {
  const trimmed = value?.replace(/\r\n?/g, '\n').trim();
  return trimmed || undefined;
};

const getMarkdownLinks = (text: string): readonly MarkdownLink[] => {
  const links: MarkdownLink[] = [];
  const seenHrefs = new Set<string>();
  const markdownLinkPattern = /(?<!!)\[([^\]\n]+)\]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/g;

  for (const match of text.matchAll(markdownLinkPattern)) {
    const label = match[1]?.trim();
    const href = match[2]?.trim();
    if (!label || !href || seenHrefs.has(href)) continue;
    seenHrefs.add(href);
    links.push({ href, markdown: `[${label}](${href})` });
  }

  for (const match of text.matchAll(/https?:\/\/[^\s<>()\]]+/g)) {
    const href = match[0].replace(/[.,;:!?]+$/, '');
    if (!href || seenHrefs.has(href)) continue;
    seenHrefs.add(href);
    links.push({ href, markdown: `[${href}](${href})` });
  }

  return links;
};

const appendDetailLinksToSummary = (
  summaryText: string,
  detailParts: readonly ThreadAssistantMessagePart[]
): string => {
  const summaryHrefs = new Set(getMarkdownLinks(summaryText).map((link) => link.href));
  const detailText = detailParts
    .filter(
      (part): part is Extract<ThreadAssistantMessagePart, { type: 'text' }> => part.type === 'text'
    )
    .map((part) => part.text)
    .join('\n');
  const missingLinks = getMarkdownLinks(detailText).filter((link) => !summaryHrefs.has(link.href));
  if (!missingLinks.length) return summaryText;

  return `${summaryText}\n\n${missingLinks.map((link) => link.markdown).join('\n\n')}`;
};

const getStudioSummaryFields = (
  blockText: string
): {
  readonly detailsLabel?: string;
  readonly summaryText?: string;
  readonly workedFor?: string;
} => {
  type StudioSummaryField = 'details_label' | 'summary' | 'title' | 'worked_for';
  const fields: Partial<Record<StudioSummaryField, string[]>> = {};

  const fieldMatches = Array.from(
    blockText.matchAll(/(?:^|\s)(title|worked_for|summary|details_label):\s*/gi)
  );

  for (const [index, match] of fieldMatches.entries()) {
    const field = match[1]!.toLowerCase() as StudioSummaryField;
    const valueStart = (match.index ?? 0) + match[0].length;
    const valueEnd = fieldMatches[index + 1]?.index ?? blockText.length;
    fields[field] = [blockText.slice(valueStart, valueEnd)];
  }

  return {
    detailsLabel: normalizeSummaryField(fields.details_label?.join(' ')),
    summaryText: normalizeMarkdownSummary(fields.summary?.join('\n')),
    workedFor: normalizeSummaryField(fields.worked_for?.join(' ')),
  };
};

const getStudioDetailsLabel = ({
  elapsedMs,
  summaryFields,
}: {
  readonly elapsedMs?: number;
  readonly summaryFields: ReturnType<typeof getStudioSummaryFields>;
}): string => {
  const elapsedDuration = formatElapsedDuration(elapsedMs);
  if (elapsedDuration) return `worked for ${elapsedDuration}`;

  if (summaryFields.detailsLabel?.toLowerCase().startsWith('worked for ')) {
    return summaryFields.detailsLabel.toLowerCase() === 'worked for unknown'
      ? ASSISTANT_WORK_DETAILS_LABEL
      : summaryFields.detailsLabel;
  }

  if (!summaryFields.workedFor || summaryFields.workedFor.toLowerCase() === 'unknown') {
    return ASSISTANT_WORK_DETAILS_LABEL;
  }

  return `worked for ${summaryFields.workedFor}`;
};

const serializeAssistantPart = (
  part: ThreadAssistantMessagePart
): SerializedAssistantPart | undefined => {
  if (part.type === 'text') {
    const text = part.text.trim();
    return text ? { type: 'text', text } : undefined;
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

const createAssistantCollapsedStudioDetailsPart = ({
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

const isAssistantSubtleToolCallPart = (part: ThreadAssistantMessagePart): boolean =>
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

const splitTextParagraphs = (text: string): readonly string[] =>
  text
    .trim()
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

const splitTrailingQuestion = (
  text: string
): { readonly detailText: string; readonly question?: string } => {
  const lines = text.trim().split('\n');
  const lastLine = lines.at(-1)?.trim();
  if (!lastLine?.endsWith('?')) return { detailText: text.trim() };

  return {
    detailText: lines.slice(0, -1).join('\n').trim(),
    question: lastLine,
  };
};

const MARKDOWN_TABLE_DELIMITER_ROW = /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/;

const extractMarkdownTableParts = (
  text: string
): {
  readonly remainingText: string;
  readonly tableParts: readonly ThreadAssistantMessagePart[];
} => {
  const lines = text.split('\n');
  const tableLines = new Set<number>();
  const tableParts: ThreadAssistantMessagePart[] = [];

  for (let index = 1; index < lines.length; index += 1) {
    if (!MARKDOWN_TABLE_DELIMITER_ROW.test(lines[index]!) || !lines[index - 1]!.includes('|')) {
      continue;
    }

    const tableStart = index - 1;
    let tableEnd = index;
    while (tableEnd + 1 < lines.length && lines[tableEnd + 1]!.trim().includes('|')) {
      tableEnd += 1;
    }

    for (let tableLine = tableStart; tableLine <= tableEnd; tableLine += 1) {
      tableLines.add(tableLine);
    }
    tableParts.push({
      type: 'text',
      text: lines
        .slice(tableStart, tableEnd + 1)
        .join('\n')
        .trim(),
    });
    index = tableEnd;
  }

  const remainingText = lines
    .filter((_, index) => !tableLines.has(index))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  return { remainingText, tableParts };
};

const isFinalSummaryToolPart = (part: ThreadAssistantMessagePart): boolean =>
  part.type === 'tool-call' &&
  part.toolName !== ASSISTANT_COLLAPSED_THINKING_TOOL_NAME &&
  part.toolName !== ASSISTANT_THINKING_TOOL_NAME &&
  part.toolName !== ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME &&
  part.toolName !== ASSISTANT_SUBTLE_TOOL_GROUP_NAME &&
  !isAssistantSubtleToolCallPart(part);

const partitionStudioSummaryParts = (
  parts: readonly ThreadAssistantMessagePart[]
): {
  readonly detailParts: readonly ThreadAssistantMessagePart[];
  readonly finalParts: readonly ThreadAssistantMessagePart[];
} => {
  const detailParts: ThreadAssistantMessagePart[] = [];
  const finalParts: ThreadAssistantMessagePart[] = [];

  for (const part of parts) {
    if (part.type === 'text') {
      const { remainingText, tableParts } = extractMarkdownTableParts(part.text);
      if (remainingText) detailParts.push({ ...part, text: remainingText });
      finalParts.push(...tableParts);
      continue;
    }

    if (isFinalSummaryToolPart(part)) {
      finalParts.push(part);
    } else {
      detailParts.push(part);
    }
  }

  return { detailParts, finalParts };
};

const getTruncatedStudioSummaryText = (
  detailParts: readonly ThreadAssistantMessagePart[],
  question: string | undefined
): string => {
  const detailText = detailParts
    .filter(
      (part): part is Extract<ThreadAssistantMessagePart, { type: 'text' }> => part.type === 'text'
    )
    .map((part) => part.text)
    .join('\n\n');
  const detailSummary = splitTextParagraphs(detailText).at(-1);
  const baseSummary = detailSummary ?? 'The response ended before its summary was complete.';
  return question && !baseSummary.includes(question)
    ? `${baseSummary}\n\n${question}`
    : baseSummary;
};

const getStudioSummaryBlock = (
  parts: readonly ThreadAssistantMessagePart[],
  options: AssistantCompletedMessageOptions
): StudioSummaryBlock | undefined => {
  const detailParts: ThreadAssistantMessagePart[] = [];

  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index]!;
    if (part.type !== 'text') {
      detailParts.push(part);
      continue;
    }

    const summaryStartIndex = part.text.indexOf(STUDIO_MESSAGE_SUMMARY_START);
    if (summaryStartIndex < 0) {
      detailParts.push(part);
      continue;
    }

    const textBeforeSummary = part.text.slice(0, summaryStartIndex).trim();
    const { detailText, question } = splitTrailingQuestion(textBeforeSummary);
    if (detailText) {
      detailParts.push({ type: 'text', text: detailText });
    }

    const summarySourceParts = [part.text.slice(summaryStartIndex)];
    for (const remainingPart of parts.slice(index + 1)) {
      if (remainingPart.type === 'text') {
        summarySourceParts.push(remainingPart.text);
      } else {
        detailParts.push(remainingPart);
      }
    }
    const summarySource = summarySourceParts.join('\n');
    const summaryEndIndex = summarySource.indexOf(STUDIO_MESSAGE_SUMMARY_END);
    if (summaryEndIndex < 0) {
      const partitionedParts = partitionStudioSummaryParts(detailParts);
      return {
        detailsLabel: getStudioDetailsLabel({
          elapsedMs: options.elapsedMs,
          summaryFields: getStudioSummaryFields(''),
        }),
        detailParts: partitionedParts.detailParts,
        finalParts: partitionedParts.finalParts,
        summaryText: getTruncatedStudioSummaryText(detailParts, question),
      };
    }

    const summaryBlockText = summarySource
      .slice(STUDIO_MESSAGE_SUMMARY_START.length, summaryEndIndex)
      .trim();
    const trailingSummaryText = summarySource
      .slice(summaryEndIndex + STUDIO_MESSAGE_SUMMARY_END.length)
      .trim();
    const summaryFields = getStudioSummaryFields(summaryBlockText);
    const baseSummaryText =
      summaryFields.summaryText ?? normalizeMarkdownSummary(trailingSummaryText);
    if (!baseSummaryText) return undefined;
    const summaryTextWithQuestion =
      question && !baseSummaryText.includes(question)
        ? `${baseSummaryText}\n\n${question}`
        : baseSummaryText;
    const summaryText = appendDetailLinksToSummary(summaryTextWithQuestion, detailParts);
    const partitionedParts = partitionStudioSummaryParts(detailParts);

    return {
      detailsLabel: getStudioDetailsLabel({
        elapsedMs: options.elapsedMs,
        summaryFields,
      }),
      detailParts: partitionedParts.detailParts,
      finalParts: partitionedParts.finalParts,
      summaryText,
    };
  }

  return undefined;
};

const getCollapsedTextParts = (
  parts: readonly ThreadAssistantMessagePart[],
  lastToolIndex: number
): {
  readonly collapsedText: string | undefined;
  readonly summaryText: string | undefined;
} => {
  const thinkingTextParts: string[] = [];
  const trailingTextParts: string[] = [];

  parts.forEach((part, index) => {
    if (isAssistantThinkingPart(part)) {
      const reasoning = getAssistantThinkingPartText(part).trim();
      if (reasoning) thinkingTextParts.push(reasoning);
      return;
    }

    if (part.type !== 'text') return;

    const text = part.text.trim();
    if (!text) return;

    if (index > lastToolIndex) {
      trailingTextParts.push(text);
    } else {
      thinkingTextParts.push(text);
    }
  });

  const trailingParagraphs = splitTextParagraphs(trailingTextParts.join('\n\n'));
  const summaryParagraphs = trailingParagraphs.slice(-2);
  const collapsedTrailingParagraphs = trailingParagraphs.slice(0, -2);
  const collapsedParagraphs = splitTextParagraphs(
    [...thinkingTextParts, collapsedTrailingParagraphs.join('\n\n')].filter(Boolean).join('\n\n')
  );

  return {
    collapsedText: collapsedParagraphs.length ? collapsedParagraphs.join('\n\n') : undefined,
    summaryText: summaryParagraphs.length ? summaryParagraphs.join('\n\n') : undefined,
  };
};

const joinCollapsedThinking = (...texts: readonly (string | undefined)[]): string | undefined => {
  const joined = texts
    .map((text) => text?.trim())
    .filter(Boolean)
    .join('\n\n');
  return joined || undefined;
};

export const getAssistantCompletedMessageParts = (
  parts: readonly ThreadAssistantMessagePart[],
  options: AssistantCompletedMessageOptions = {}
): readonly ThreadAssistantMessagePart[] => {
  const reasoningText = joinCollapsedThinking(...parts.map(getAssistantThinkingPartText));
  const remainingParts = parts.filter((part) => !isAssistantThinkingPart(part));

  const studioSummaryBlock = getStudioSummaryBlock(remainingParts, options);
  if (studioSummaryBlock) {
    const completedParts: ThreadAssistantMessagePart[] = [];
    if (reasoningText) {
      completedParts.push(createAssistantCollapsedThinkingPart(reasoningText));
    }
    const collapsedDetailsPart = createAssistantCollapsedStudioDetailsPart({
      label: studioSummaryBlock.detailsLabel,
      parts: studioSummaryBlock.detailParts,
    });
    if (collapsedDetailsPart) completedParts.push(collapsedDetailsPart);
    completedParts.push(...studioSummaryBlock.finalParts);
    completedParts.push({ type: 'text', text: studioSummaryBlock.summaryText });
    return completedParts;
  }

  const completedParts: ThreadAssistantMessagePart[] = [];
  let lastToolIndex = -1;

  parts.forEach((part, index) => {
    if (isAssistantThinkingPart(part)) return;
    if (part.type === 'tool-call') {
      completedParts.push(part);
      lastToolIndex = index;
    }
  });

  if (lastToolIndex < 0) {
    if (!reasoningText) return remainingParts;
    return [createAssistantCollapsedThinkingPart(reasoningText), ...remainingParts];
  }

  const { collapsedText, summaryText } = getCollapsedTextParts(parts, lastToolIndex);
  if (collapsedText) completedParts.unshift(createAssistantCollapsedThinkingPart(collapsedText));
  for (const part of parts.slice(lastToolIndex + 1)) {
    if (part.type === 'text' || isAssistantThinkingPart(part)) continue;
    completedParts.push(part);
  }
  if (summaryText) completedParts.push({ type: 'text', text: summaryText });

  return completedParts;
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
