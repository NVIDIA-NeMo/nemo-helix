// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SamplerType } from '@nemo/sdk/generated/data-designer/schema/SamplerType';
import {
  type BuilderColumn,
  getSeedAvailableColumns,
  SEED_FILESET_REF_KEY,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';

/** A short, human summary of a column for the schema-list renderer. */
export interface ColumnDescription {
  /** Badge label for the column's type (or sampler sub-type). */
  typeLabel: string;
  /** One-line summary of what the column produces, or `null` if it's not yet configured. */
  detail: string | null;
}

/** Splits a comma-separated field value into trimmed, non-empty entries. */
const splitList = (value: string | undefined): string[] =>
  (value ?? '')
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);

/** `${n} values · a, b, c, …`, collapsing to `a / b / c` when there are only a few. */
const summarizeValues = (values: string[]): string => {
  if (values.length === 0) return 'no values yet';
  if (values.length <= 3) return values.join(' / ');
  return `${values.length} values · ${values.slice(0, 4).join(', ')}, …`;
};

/** Names pulled from an `llm-judge` column's `scores` JSON array, if it parses. */
const parseScoreNames = (scores: string | undefined): string[] => {
  if (!scores?.trim()) return [];
  try {
    const parsed: unknown = JSON.parse(scores);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((score) =>
        score && typeof score === 'object' ? (score as { name?: unknown }).name : undefined
      )
      .filter((name): name is string => typeof name === 'string' && name.length > 0);
  } catch {
    return [];
  }
};

/**
 * Produces the badge label and summary line shown for a column in the schema list. The
 * summary is intentionally terse — relationships (referenced columns) are surfaced
 * separately as tags on the row.
 */
export const describeColumn = (column: BuilderColumn): ColumnDescription => {
  const { option, values } = column;

  const detail = ((): string | null => {
    switch (option.columnType) {
      case 'sampler':
        if (option.samplerType === SamplerType.category) {
          return summarizeValues(splitList(values.values));
        }
        return option.description;
      case 'llm-text':
        return 'generator';
      case 'llm-code':
        return values.code_lang ? `code generator · ${values.code_lang}` : 'code generator';
      case 'llm-structured':
        return 'structured generator';
      case 'image':
        return 'image generator';
      case 'llm-judge': {
        const names = parseScoreNames(values.scores);
        if (names.length === 0) return 'judge';
        return `judge · ${names.length} ${names.length === 1 ? 'score' : 'scores'}: ${names.join(', ')}`;
      }
      case 'embedding':
        return values.target_column ? `embedding of ${values.target_column}` : 'embedding';
      case 'expression':
        return 'expression · no LLM';
      case 'validation':
        return values.validator_type ? `validation · ${values.validator_type}` : 'validation';
      case 'seed-dataset': {
        const fileset = values[SEED_FILESET_REF_KEY]?.trim();
        const columns = getSeedAvailableColumns(column);
        if (fileset)
          return `seed · ${fileset}${columns.length ? ` · ${columns.length} columns` : ''}`;
        return 'seed dataset';
      }
      case 'custom':
        return 'custom function';
      default:
        return option.description;
    }
  })();

  return { typeLabel: option.label, detail };
};
