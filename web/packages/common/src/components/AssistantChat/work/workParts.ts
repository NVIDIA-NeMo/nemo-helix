// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CodeSnippetLanguage } from '@nvidia/foundations-react-core';

const JSON_NUMBER_PATTERN = /-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;

const numberTokens = (text: string): string => (text.match(JSON_NUMBER_PATTERN) ?? []).join(',');

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
