// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jsonParseLinter } from '@codemirror/lang-json';
import { type Diagnostic, linter } from '@codemirror/lint';
import type { EditorView } from '@codemirror/view';
import { findNodeRanges, type NodeRange } from '@nemo/common/src/components/CodeEditor/util/editor';

// JSONL linter
export const jsonlLinter = linter((view: EditorView) => {
  const content = view.state.doc.toString();
  const diagnostics: Diagnostic[] = [];

  try {
    const ranges = findNodeRanges(content);

    // If no valid ranges found but content exists, mark entire content as invalid
    if (ranges.length === 0 && content.trim()) {
      diagnostics.push({
        from: 0,
        to: content.length,
        severity: 'error',
        message: 'Invalid JSONL: No valid JSON objects found',
      });
      return diagnostics;
    }

    ranges.forEach((range: NodeRange) => {
      const node = content.slice(range.start, range.end);
      try {
        const parsed = JSON.parse(node);
        // Validate it's an object (not array/null/primitive)
        if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
          diagnostics.push({
            from: range.start,
            to: range.end,
            severity: 'error',
            message: 'Invalid JSONL: Each line must be a JSON object',
          });
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        diagnostics.push({
          from: range.start,
          to: range.end,
          severity: 'error',
          message: `Invalid JSON object: ${errorMessage}`,
        });
      }
    });
  } catch {
    // If findNodeRanges fails, mark entire content as invalid
    diagnostics.push({
      from: 0,
      to: content.length,
      severity: 'error',
      message: 'Invalid JSONL structure',
    });
  }

  return diagnostics;
});

// JSON linter - use the built-in jsonParseLinter
export const jsonLinter = linter((view: EditorView) => {
  return jsonParseLinter()(view);
});
