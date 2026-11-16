// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Diagnostic, linter } from '@codemirror/lint';

export const yamlLinter = linter(async (view) => {
  const diagnostics: Diagnostic[] = [];
  const { default: YAML, YAMLParseError } = await import('yaml');

  try {
    YAML.parse(view.state.doc.toString());
  } catch (error: unknown) {
    if (error instanceof YAMLParseError) {
      const loc = error.pos;
      const from = loc ? loc[0] : 0;
      const to = loc ? loc[1] : 0;
      const severity = 'error';

      diagnostics.push({
        from,
        to,
        message: error.message,
        severity,
      });
    }
  }
  return diagnostics;
});
