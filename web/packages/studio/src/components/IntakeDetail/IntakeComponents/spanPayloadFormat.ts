// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  toChatMessages,
  type ChatMessage,
} from '@studio/components/IntakeDetail/IntakeComponents/openaiChat';

export type SpanPayloadFormat = 'raw' | 'md' | 'json' | 'chat';

export interface SpanPayloadFormatState {
  format: SpanPayloadFormat;
  select: (format: SpanPayloadFormat) => void;
  isJson: boolean;
  isChat: boolean;
  isEmpty: boolean;
}

// TypeScript's lib does not declare JSON source text access yet.
type SourceTextReviver = (key: string, value: unknown, context?: { source?: string }) => unknown;
const rawJSON = (JSON as typeof JSON & { rawJSON?: (text: string) => unknown }).rawJSON;

// Re-emits each number as the literal it was parsed from, so an int64 span id
// such as 9007199254740993 is not rounded to ...992 by a float64 round trip,
// and 1.0, 1e2, and -0 keep the form they were written in. Engines without
// JSON source text access pass no context and fall through to the parsed value.
const keepNumberSource: SourceTextReviver = (_key, value, context) =>
  rawJSON && typeof value === 'number' && context?.source !== undefined
    ? rawJSON(context.source)
    : value;

// Objects and arrays are never null, so null means "not a JSON payload".
const parseJson = (value: string | null | undefined): unknown => {
  const trimmed = value?.trim();
  if (!trimmed || !(trimmed.startsWith('{') || trimmed.startsWith('['))) {
    return null;
  }
  try {
    return JSON.parse(trimmed, keepNumberSource as (key: string, value: unknown) => unknown);
  } catch {
    return null;
  }
};

/** The chat turns in `value`, or `null` when it is not an OpenAI-compatible payload. */
export const parseChatPayload = (value: string | null | undefined): ChatMessage[] | null =>
  toChatMessages(parseJson(value));

/** Which views a payload supports, from a single parse. */
export const readPayloadFormats = (
  value: string | null | undefined
): { isJson: boolean; isChat: boolean } => {
  const parsed = parseJson(value);
  return { isJson: parsed !== null, isChat: toChatMessages(parsed) !== null };
};

/**
 * Pretty-printed `value` when it is a JSON object or array, else `null`. A
 * payload that repeats a key keeps only the last, which the reviver cannot
 * reach — `raw` remains the exact view.
 */
export const parseJsonPayload = (value: string | null | undefined): string | null => {
  const parsed = parseJson(value);
  return parsed === null ? null : JSON.stringify(parsed, null, 2);
};

export const autoFormat = (isJson: boolean, isChat = false): SpanPayloadFormat =>
  isChat ? 'chat' : isJson ? 'json' : 'raw';
