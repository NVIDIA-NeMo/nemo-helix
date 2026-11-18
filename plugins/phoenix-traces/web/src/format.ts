// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parsePartialJson } from './partialJson';

export interface Message {
  role?: string;
  text: string;
}

export interface ReviewGroup {
  title: string;
  attention?: string;
  description?: string;
  diff?: string;
}

export interface StructuredOutput {
  fields: Record<string, unknown>;
  groups: ReviewGroup[];
  /** The exporter cut the output; what is shown is what arrived. */
  partial: boolean;
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

/**
 * Span input/output arrive as a plain string, a `{role, text}` message, or a list of
 * those, depending on the span kind. Normalize to messages.
 */
export const readMessages = (raw: unknown): Message[] => {
  if (typeof raw !== 'string') return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [{ text: raw }];
  }
  const items = Array.isArray(parsed) ? parsed : [parsed];
  const messages = items.flatMap((item): Message[] => {
    if (!isRecord(item)) return [];
    const text = typeof item.text === 'string' ? item.text : typeof item.content === 'string' ? item.content : undefined;
    return text === undefined ? [] : [{ role: typeof item.role === 'string' ? item.role : undefined, text }];
  });
  return messages.length > 0 ? messages : [{ text: raw }];
};

/** `{role, text, tool_calls}` messages can carry tool calls next to (or instead of) text. */
export const readToolCalls = (raw: unknown): unknown[] => {
  if (typeof raw !== 'string') return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    return isRecord(parsed) && Array.isArray(parsed.tool_calls) ? parsed.tool_calls : [];
  } catch {
    return [];
  }
};

/** The final message's text parsed as a review object, if that is what it holds. */
export const readStructured = (raw: unknown): StructuredOutput | undefined => {
  const messages = readMessages(raw);
  const text = messages[messages.length - 1]?.text.trim();
  if (!text?.startsWith('{')) return undefined;
  const parsed = parsePartialJson(text);
  if (!parsed || !isRecord(parsed.value) || !Array.isArray(parsed.value.groups)) return undefined;
  const { groups, ...fields } = parsed.value;
  return {
    fields,
    groups: groups.filter(isRecord).map((g) => ({
      title: typeof g.title === 'string' ? g.title : 'Untitled group',
      attention: typeof g.attention === 'string' ? g.attention : undefined,
      description: typeof g.description === 'string' ? g.description : undefined,
      diff: typeof g.diff === 'string' ? g.diff : undefined,
    })),
    partial: parsed.partial,
  };
};

/** `blast_radius_reason` -> `Blast radius reason`. */
export const humanize = (key: string): string => {
  const words = key.replace(/[_-]+/g, ' ').trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
};
