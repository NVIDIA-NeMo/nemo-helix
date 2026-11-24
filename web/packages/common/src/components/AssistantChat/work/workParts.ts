// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
  ASSISTANT_UNKNOWN_WORK_DURATION_LABEL,
  ASSISTANT_WORK_DETAILS_LABEL,
  isAssistantJobProgressToolName,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import { isPlainObject } from '@nemo/common/src/utils/isPlainObject';
import type { CodeSnippetLanguage } from '@nvidia/foundations-react-core';

const JSON_NUMBER_PATTERN = /-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;

const numberTokens = (text: string): string => (text.match(JSON_NUMBER_PATTERN) ?? []).join(',');

export const WORK_GROUP_KEY = 'group-work';

interface WorkPartLike {
  readonly type: string;
  readonly toolName?: string;
  readonly args?: unknown;
}

// Job progress cards stay visible next to the answer instead of hiding in the work group.
const isWorkPart = (part: WorkPartLike): boolean => {
  if (part.type === 'reasoning') {
    return true;
  }
  return part.type === 'tool-call' && !isAssistantJobProgressToolName(part.toolName ?? '');
};

export const groupWorkParts = (part: WorkPartLike): readonly [typeof WORK_GROUP_KEY] | null => {
  if (isWorkPart(part)) {
    return [WORK_GROUP_KEY];
  }
  return null;
};

const toWorkPart = (value: unknown): WorkPartLike | undefined => {
  if (!isPlainObject(value) || typeof value.type !== 'string') {
    return undefined;
  }
  return {
    type: value.type,
    toolName: typeof value.toolName === 'string' ? value.toolName : undefined,
    args: value.args,
  };
};

const computeToolCallCount = (part: WorkPartLike): number => {
  if (part.type !== 'tool-call') {
    return 0;
  }
  const args = isPlainObject(part.args) ? part.args : {};

  switch (part.toolName) {
    case ASSISTANT_THINKING_TOOL_NAME:
    case ASSISTANT_COLLAPSED_THINKING_TOOL_NAME:
      return 0;
    case ASSISTANT_SUBTLE_TOOL_GROUP_NAME:
      return Array.isArray(args.actions) ? args.actions.length : 0;
    case ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME: {
      if (!Array.isArray(args.parts)) {
        return 0;
      }
      return args.parts.reduce<number>((total, nested) => {
        const nestedPart = toWorkPart(nested);
        return total + (nestedPart ? computeToolCallCount(nestedPart) : 0);
      }, 0);
    }
    default:
      return 1;
  }
};

const toolCallCountCache = new WeakMap<WorkPartLike, number>();

/** The real tool calls a part stands for, unpacking the NeMo Assistant's synthetic parts. */
export const countToolCalls = (part: WorkPartLike): number => {
  const cached = toolCallCountCache.get(part);
  if (cached !== undefined) {
    return cached;
  }
  const count = computeToolCallCount(part);
  toolCallCountCache.set(part, count);
  return count;
};

const startsWorkGroup = (parts: readonly WorkPartLike[], index: number): boolean => {
  const part = parts[index];
  if (!part || !isWorkPart(part)) {
    return false;
  }
  const previous = parts[index - 1];
  return !previous || !isWorkPart(previous);
};

export const getWorkGroupOrdinal = (parts: readonly WorkPartLike[], firstIndex: number): number => {
  let ordinal = 0;
  for (let index = 0; index < firstIndex; index += 1) {
    if (startsWorkGroup(parts, index)) {
      ordinal += 1;
    }
  }
  return ordinal;
};

export const countGroupToolCalls = (
  parts: readonly WorkPartLike[],
  indices: readonly number[]
): number =>
  indices.reduce((total, index) => {
    const part = parts[index];
    return total + (part ? countToolCalls(part) : 0);
  }, 0);

/** The "worked for 42s" label the NeMo Assistant stores on its collapsed work details. */
export const getWorkedForLabel = (
  parts: readonly WorkPartLike[],
  indices: readonly number[]
): string | undefined => {
  for (const index of indices) {
    const part = parts[index];
    if (part?.toolName !== ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME) {
      continue;
    }
    const label = isPlainObject(part.args) ? part.args.label : undefined;
    if (typeof label !== 'string') {
      continue;
    }
    const trimmed = label.trim();
    if (
      !trimmed ||
      trimmed === ASSISTANT_WORK_DETAILS_LABEL ||
      trimmed.toLowerCase() === ASSISTANT_UNKNOWN_WORK_DURATION_LABEL
    ) {
      continue;
    }
    return trimmed;
  }
  return undefined;
};

export const getWorkGroupLabel = ({
  active,
  toolCallCount,
  workedFor,
}: {
  readonly active: boolean;
  readonly toolCallCount: number;
  readonly workedFor?: string;
}): string => {
  if (active) {
    return 'Thinking…';
  }
  const summary =
    toolCallCount > 0
      ? `Ran ${toolCallCount} tool call${toolCallCount === 1 ? '' : 's'}`
      : 'Thought';
  if (!workedFor) {
    return summary;
  }
  return `${summary} · ${workedFor}`;
};

export const formatToolPayload = (
  value: unknown
): { readonly language: CodeSnippetLanguage; readonly text: string } => {
  if (typeof value === 'string') {
    try {
      const formatted = JSON.stringify(JSON.parse(value), null, 2);
      // JSON.parse rounds big integers and rewrites literals like 1.0, so show those payloads as sent.
      if (numberTokens(formatted) !== numberTokens(value)) {
        return { language: 'json', text: value };
      }
      return { language: 'json', text: formatted };
    } catch {
      return { language: 'text', text: value };
    }
  }
  return { language: 'json', text: JSON.stringify(value, null, 2) ?? String(value) };
};
