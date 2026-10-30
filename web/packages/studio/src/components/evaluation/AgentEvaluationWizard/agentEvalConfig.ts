// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { buildMetricBundles } from '@studio/components/evaluation/EvalWizard/buildEvaluationSpec';
import { ALWAYS_AVAILABLE } from '@studio/components/evaluation/EvalWizard/judgePromptVariables';
import {
  type DatasetBindings,
  EVALUATION_FORM_DEFAULTS,
  type EvaluationFormValues,
} from '@studio/components/evaluation/EvalWizard/types';
import type { DatasetEvalSpec } from '@studio/components/evaluation/submitEvaluationJob';

const IDENTIFIER = /^[A-Za-z_]\w*$/;

export const itemRef = (column: string): string =>
  IDENTIFIER.test(column) ? `item.${column}` : `item['${column}']`;

const fieldBlocks = (columns: string[]) =>
  columns.map((column) => `${column}:\n{{ ${itemRef(column)} }}`);

/** The agent's prompt until the user edits it: every column, under its name. */
export const defaultAgentPrompt = (columns: string[]): string => fieldBlocks(columns).join('\n\n');

/** The judge's prompt until the user edits it: every column, then the agent's response. */
export const defaultJudgePrompt = (columns: string[]): string =>
  [...fieldBlocks(columns), 'Response to evaluate:\n{{ sample.output_text }}'].join('\n\n');

/** A judge given its prompt reads no bindings, and the judge is the only metric built here. */
const NO_BINDINGS: DatasetBindings = {
  messagesColumn: null,
  input: '',
  reference: null,
  context: null,
  inputPath: null,
  referencePath: null,
};

export type AgentJudgeValues = Pick<EvaluationFormValues['body'], 'judgeModel' | 'scores'> & {
  judgePrompt: string;
};

/** The config the guided form writes: a text prompt over the row, scored by one LLM judge. */
export const buildAgentEvaluationSpec = (
  dataset: string,
  prompt: string,
  judge: AgentJudgeValues,
  workspace: string
): DatasetEvalSpec => ({
  dataset,
  prompt_template: prompt,
  metrics: buildMetricBundles(
    { ...EVALUATION_FORM_DEFAULTS, body: { ...EVALUATION_FORM_DEFAULTS.body, ...judge } },
    NO_BINDINGS,
    workspace
  ),
});

const BLOCK_RE = /\{\{([\s\S]*?)\}\}|\{%-?([\s\S]*?)-?%\}/g;
const COMMENT_RE = /\{#[\s\S]*?#\}/g;
const STRING_RE = /'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"/g;
const ITEM_KEY_RE = /\bitem\s*\[\s*(['"])(.*?)\1\s*\]/g;
const ITEM_ATTR_RE = /\bitem\s*\.\s*([A-Za-z_]\w*)/g;
const FOR_RE = /^\s*for\s+([\w\s,]+?)\s+in\s/;
const SET_RE = /^\s*set\s+([A-Za-z_]\w*)/;
const IDENTIFIER_RE = /(?<![\w.])[A-Za-z_]\w*/g;

/** Jinja syntax and globals that read like names but never come from the row. */
const JINJA_WORDS = new Set([
  'and',
  'or',
  'not',
  'in',
  'is',
  'if',
  'else',
  'elif',
  'endif',
  'for',
  'endfor',
  'set',
  'endset',
  'true',
  'false',
  'none',
  'True',
  'False',
  'None',
  'loop',
  'range',
  'dict',
  'namespace',
  'recursive',
  'with',
  'endwith',
  'filter',
  'endfilter',
]);

/** The row fields one template reads: ``item.X`` / ``item['X']`` and bare names, minus
 *  loop and ``set`` locals, filters, tests and keywords.
 *  ponytail: regex approximation of the SDK's Jinja walk
 *  (``nhx_evals_sdk/dataset_schemas/templates.py``); macros and call keyword arguments are
 *  not handled. */
export const templateFields = (template: string): string[] => {
  const blocks = [...template.replace(COMMENT_RE, '').matchAll(BLOCK_RE)].map(
    (match) => match[1] ?? match[2] ?? ''
  );
  const locals = new Set<string>();
  for (const block of blocks) {
    FOR_RE.exec(block)?.[1]
      .split(',')
      .forEach((name) => locals.add(name.trim()));
    const set = SET_RE.exec(block)?.[1];
    if (set) locals.add(set);
  }

  const fields = new Set<string>();
  for (const block of blocks) {
    for (const match of block.matchAll(ITEM_KEY_RE)) fields.add(match[2]);
    const code = block
      .replace(STRING_RE, ' ')
      .replace(/^\s*(?:for|set)\b/, ' ')
      .replace(/\|\s*[A-Za-z_]\w*/g, '|')
      .replace(/\bis\s+(?:not\s+)?[A-Za-z_]\w*/g, ' ');
    for (const match of code.matchAll(ITEM_ATTR_RE)) fields.add(match[1]);
    for (const [name] of code.matchAll(IDENTIFIER_RE)) {
      if (!JINJA_WORDS.has(name) && !locals.has(name) && !ALWAYS_AVAILABLE.includes(name)) {
        fields.add(name);
      }
    }
  }
  return [...fields];
};

/** Every string under ``value`` except a judge's ``system_prompt``, which is sent verbatim. */
const templateStrings = (value: unknown): string[] => {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.flatMap(templateStrings);
  if (typeof value === 'object' && value !== null) {
    return Object.entries(value).flatMap(([key, entry]) =>
      key === 'system_prompt' ? [] : templateStrings(entry)
    );
  }
  return [];
};

/** Why ``spec`` cannot render against a dataset with ``columns``, or ``[]`` when it can. A name
 *  resolves when it is a column or a ``field_mapping`` key, the namespace the evaluator renders
 *  with; a mapping resolves when its path starts at a column. */
export const configFitIssues = (spec: DatasetEvalSpec, columns: string[]): string[] => {
  const columnSet = new Set(columns);
  const mapping = spec.field_mapping ?? {};
  const issues: string[] = [];

  const prompt = spec.prompt_template;
  if (
    typeof prompt === 'object' &&
    prompt !== null &&
    !('messages' in prompt || 'prompt' in prompt)
  ) {
    issues.push('The prompt template must be text or a chat "messages" list.');
  }

  for (const [field, path] of Object.entries(mapping)) {
    const column = typeof path === 'string' ? path.split(/[.[]/)[0] : '';
    if (!columnSet.has(column)) {
      issues.push(
        `Maps "${field}" to "${String(path)}", but the dataset has no "${column}" column.`
      );
    }
  }

  const templates = [
    ...templateStrings(prompt),
    ...spec.metrics.flatMap((metric) => templateStrings(metric.payload.metric)),
  ];
  const missing = new Set(
    templates.flatMap(templateFields).filter((name) => !columnSet.has(name) && !(name in mapping))
  );
  for (const name of missing) {
    issues.push(`Reads "${name}", which is not a dataset column or a mapped field.`);
  }
  return issues;
};
