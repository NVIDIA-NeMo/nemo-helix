// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CANONICAL_FIELDS,
  type CanonicalField,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';

/** Names the render context carries regardless of the dataset. */
const ALWAYS_AVAILABLE = ['item', 'sample', 'scores', 'output_text', 'response'];

export interface JudgePromptVariableReport {
  invalid: string[];
  nonPortable: { token: string; canonical: CanonicalField }[];
  malformed: boolean;
}

const TOKEN_RE = /\{\{\s*([^}]*?)\s*\}\}/g;

/** The leading identifier of a template expression, or null when it does not
 *  begin with one. */
export const expressionRoot = (expression: string): string | null => {
  const root = /^\s*([A-Za-z_]\w*)/.exec(expression);
  return root ? root[1] : null;
};

/** Every `{{ ... }}` body in the template. `{% ... %}` blocks are out of scope. */
const templateExpressions = (template: string): string[] =>
  [...template.matchAll(TOKEN_RE)].map((match) => match[1]);

/** Braces that nest or do not pair up, which Jinja cannot parse. */
const hasMalformedBraces = (template: string): boolean =>
  /\{\{|\}\}/.test(template.replace(TOKEN_RE, '')) ||
  templateExpressions(template).some((body) => /[{}]/.test(body));

/**
 * Classifies the variables a judge prompt uses.
 *
 * A mapped column renders as well as its canonical name does, but pins the saved
 * configuration to one dataset, so it is reported apart from a name that does not
 * resolve at all.
 */
export const classifyJudgePromptVariables = (
  template: string,
  columns: string[],
  fieldMapping: Partial<Record<CanonicalField, string>>
): JudgePromptVariableReport => {
  const canonicalByColumn = new Map<string, CanonicalField>();
  for (const field of CANONICAL_FIELDS) {
    const column = fieldMapping[field];
    if (column) canonicalByColumn.set(column, field);
  }

  const columnSet = new Set(columns);
  const invalid: string[] = [];
  const nonPortable: { token: string; canonical: CanonicalField }[] = [];

  for (const expression of templateExpressions(template)) {
    const root = expressionRoot(expression);
    if (!root) continue;
    if (ALWAYS_AVAILABLE.includes(root)) continue;
    if ((CANONICAL_FIELDS as readonly string[]).includes(root)) continue;

    const canonical = canonicalByColumn.get(root);
    if (canonical) {
      if (!nonPortable.some((entry) => entry.token === root)) {
        nonPortable.push({ token: root, canonical });
      }
      continue;
    }
    if (columnSet.has(root)) continue;
    if (!invalid.includes(root)) invalid.push(root);
  }

  return { invalid, nonPortable, malformed: hasMalformedBraces(template) };
};

/** Shared by the editor's highlighting and the step gate. */
export const isKnownJudgeVariable =
  (columns: string[], fieldMapping: Partial<Record<CanonicalField, string>>) =>
  (token: string): boolean =>
    classifyJudgePromptVariables(`{{${token}}}`, columns, fieldMapping).invalid.length === 0;
