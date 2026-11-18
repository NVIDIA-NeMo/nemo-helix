// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Span attributes are cut at a size limit by the exporter ("… [truncated N chars]"), so a
// long JSON output usually arrives unterminated. Strict JSON.parse then fails and the user
// sees an unreadable wall of text. This closes the open string/brackets and parses what is there.

const close = (text: string): string => {
  const stack: string[] = [];
  let inString = false;
  let escaped = false;
  for (const ch of text) {
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') stack.push('}');
    else if (ch === '[') stack.push(']');
    else if (ch === '}' || ch === ']') stack.pop();
  }
  let out = text;
  if (inString) {
    if (escaped) out = out.slice(0, -1); // dangling backslash
    out = out.replace(/\\u[0-9a-fA-F]{0,3}$/, '') + '"'; // partial \uXXXX escape
  }
  return out.replace(/[,:\s]+$/, '') + stack.reverse().join('');
};

export interface ParsedJson {
  value: unknown;
  /** True when the text was cut off and the value is only what arrived. */
  partial: boolean;
}

export const parsePartialJson = (text: string): ParsedJson | undefined => {
  try {
    return { value: JSON.parse(text), partial: false };
  } catch {
    // fall through to repair
  }
  // A cut can land after a bare key (`"title"`) which no closing fixes: drop back to the
  // previous comma and retry a few times.
  let cut = text;
  for (let attempt = 0; attempt < 4 && cut; attempt++) {
    try {
      return { value: JSON.parse(close(cut)), partial: true };
    } catch {
      cut = cut.slice(0, Math.max(cut.lastIndexOf(','), 0));
    }
  }
  return undefined;
};
