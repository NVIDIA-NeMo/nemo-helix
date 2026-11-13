// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CodeSnippetLanguage } from '@nvidia/foundations-react-core';

export const formatToolPayload = (
  value: unknown
): { readonly language: CodeSnippetLanguage; readonly text: string } => {
  if (typeof value === 'string') {
    try {
      return { language: 'json', text: JSON.stringify(JSON.parse(value), null, 2) };
    } catch {
      return { language: 'text', text: value };
    }
  }
  return { language: 'json', text: JSON.stringify(value, null, 2) ?? String(value) };
};
