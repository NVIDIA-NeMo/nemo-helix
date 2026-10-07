// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import {
  evaluationFilesetName,
  findEvalConfigFile,
} from '@studio/components/evaluation/experimentEvalConfig';
import {
  type DatasetEvalSpec,
  type EvalSpec,
  isDatasetEvalSpec,
  parseEvalConfig,
  type PersistedEvalSpec,
} from '@studio/components/evaluation/submitEvaluationJob';
import {
  renderPromptTemplate,
  UnsupportedTemplateError,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/promptTemplate';
import { asRecord } from '@studio/util/guards';
import Papa from 'papaparse';

/** Optimize job input: the agent's question and the judge's expected answer. */
export interface StudyRow {
  id: string;
  question: string;
  answer: string;
}

/** The evaluator's canonical field for the expected answer: the row column of that name, or the
 *  column `field_mapping.reference` binds to it. */
const REFERENCE_FIELD = 'reference';

/** `workspace/fileset#path`, the dataset ref Studio bakes into a dataset-driven config. */
const DATASET_REF = /^([\w\-.]+)\/([\w\-.]+)#(.+)$/;

const asText = (value: unknown): string | undefined => {
  if (value === undefined || value === null) return undefined;
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  return text.trim() ? text : undefined;
};

const lookup = (record: Record<string, unknown>, path: string): unknown =>
  path.split('.').reduce<unknown>((value, key) => asRecord(value)?.[key], record);

/** Add canonical and custom bindings alongside the original columns, as the evaluator does. */
const applyFieldMapping = (
  record: Record<string, unknown>,
  fieldMapping: DatasetEvalSpec['field_mapping']
): Record<string, unknown> => {
  if (!fieldMapping) return record;
  const { custom, ...known } = fieldMapping;
  const mapped = { ...record };
  for (const [canonical, path] of Object.entries({ ...known, ...asRecord(custom) })) {
    if (typeof path !== 'string') continue;
    const value = lookup(record, path);
    if (value !== undefined) mapped[canonical] = value;
  }
  return mapped;
};

/** The evaluator sends a prompt directly, or the last user message of a chat template. */
const agentInputTemplate = (template: DatasetEvalSpec['prompt_template']): string | undefined => {
  if (template === undefined || typeof template === 'string') return template;
  if (typeof template.prompt === 'string') return template.prompt;
  const messages = Array.isArray(template.messages) ? template.messages : [];
  const user = messages.map(asRecord).findLast((message) => message?.role === 'user');
  return typeof user?.content === 'string' ? user.content : undefined;
};

/** Formats the evaluator's loader reads that Studio cannot parse in the browser. */
const BINARY_DATASET = /\.(parquet|feather|arrow|orc|gz|gzip)$/i;

const parseCsvRecords = (text: string, path: string): Record<string, unknown>[] => {
  const { data, errors } = Papa.parse<Record<string, string>>(text, {
    header: true,
    skipEmptyLines: true,
  });
  const [error] = errors;
  if (error) {
    throw new Error(`Could not parse ${path} as CSV: ${error.message} (row ${error.row}).`);
  }
  return data;
};

/** JSON array, JSON Lines or CSV — the text formats the evaluator's loader accepts. The format is
 *  read from the extension as the loader does, except that anything not CSV is tried as JSON. */
export const parseDatasetRecords = (text: string, path = ''): Record<string, unknown>[] => {
  if (/\.csv$/i.test(path)) return parseCsvRecords(text, path);
  if (BINARY_DATASET.test(path)) {
    throw new Error(
      `The evaluation's dataset ${path} is not JSON, JSON Lines or CSV, so the study cannot stage it.`
    );
  }
  const trimmed = text.trim();
  let records: unknown[];
  try {
    const parsed: unknown = JSON.parse(trimmed);
    records = Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    records = text.split('\n').flatMap((line, index) => {
      if (!line.trim()) return [];
      try {
        return [JSON.parse(line)];
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        throw new Error(`Could not parse ${path || 'the dataset'} line ${index + 1}: ${reason}`);
      }
    });
  }
  return records.map((value) => {
    const record = asRecord(value);
    if (!record) throw new Error('Every dataset record must be a JSON object.');
    return record;
  });
};

/** The judge scores against the expected answer, so a row without one would score as noise. */
const requireAnswer = (answer: string | undefined, missing: string): string => {
  if (answer === undefined) throw new Error(missing);
  return answer;
};

/** A task's reference is its whole ground truth; a single entry is the answer itself. */
const taskAnswer = (reference: Record<string, unknown>): string | undefined => {
  const values = Object.values(reference);
  if (values.length === 0) return undefined;
  return asText(values.length === 1 ? values[0] : reference);
};

const taskRows = (spec: PersistedEvalSpec): StudyRow[] =>
  spec.tasks.map(({ id, inputs, intent, reference = {} }) => ({
    id,
    question: inputs?.instruction || intent,
    answer: requireAnswer(
      taskAnswer(reference),
      `Task ${id} has no reference answer to score trials against.`
    ),
  }));

const datasetRows = (spec: DatasetEvalSpec, records: Record<string, unknown>[]): StudyRow[] => {
  if (spec.prompt_template === undefined) {
    throw new Error('The evaluation has no prompt template, so the study has no prompt to send.');
  }
  const template = agentInputTemplate(spec.prompt_template);
  if (template === undefined) {
    throw new Error(
      "The evaluation's prompt template has no prompt or user message the study can send."
    );
  }
  const referencePath = asRecord(spec.field_mapping)?.[REFERENCE_FIELD];
  const referenceColumn = typeof referencePath === 'string' ? referencePath : REFERENCE_FIELD;

  return records.map((raw, index) => {
    const record = applyFieldMapping(raw, spec.field_mapping);
    const id = asText(record.id) ?? String(index);
    let question: string;
    try {
      question = renderPromptTemplate(template, record);
    } catch (error) {
      if (!(error instanceof UnsupportedTemplateError)) throw error;
      throw new Error(
        `Could not render the evaluation's prompt template for dataset row ${id}: ` +
          `${error.message} is not supported here.`
      );
    }
    if (!question.trim()) {
      throw new Error(
        `Dataset row ${id} has no prompt: the evaluation's prompt template renders empty for it.`
      );
    }
    return {
      id,
      question,
      answer: requireAnswer(
        asText(record[REFERENCE_FIELD]),
        `Dataset row ${id} has no expected answer to score trials against: the evaluation reads ` +
          `it from "${referenceColumn}", which the row does not carry. Set ` +
          `field_mapping.reference to the dataset's answer column.`
      ),
    };
  });
};

/** Resolve inline tasks, inline dataset records, or a `workspace/fileset#path` dataset. */
export const studyRowsFromSpec = async (
  spec: EvalSpec,
  readDataset: (workspace: string, fileset: string, path: string) => Promise<string>
): Promise<StudyRow[]> => {
  if (!isDatasetEvalSpec(spec)) return taskRows(spec);
  if (Array.isArray(spec.dataset)) return datasetRows(spec, spec.dataset);

  const ref = DATASET_REF.exec(spec.dataset);
  if (!ref) {
    throw new Error(
      `The evaluation's dataset "${spec.dataset}" is not a workspace/fileset#path ref, so the ` +
        'study cannot stage it.'
    );
  }
  const [, workspace, fileset, path] = ref;
  return datasetRows(spec, parseDatasetRecords(await readDataset(workspace, fileset, path), path));
};

const downloadText = async (workspace: string, fileset: string, path: string): Promise<string> => {
  const blob = await filesDownloadFile(workspace, fileset, path);
  if (!blob) throw new Error(`Could not read ${path} from fileset "${fileset}".`);
  return blob.text();
};

/** Intake omits expected answers; load rows from Studio's stored eval config instead. */
export const loadStudyRows = async (
  workspace: string,
  evaluation: EvaluationResponse
): Promise<StudyRow[]> => {
  const filesetName = evaluationFilesetName(evaluation);
  if (!filesetName) {
    throw new Error(
      `Evaluation "${evaluation.name}" has no stored eval config, so there are no expected ` +
        'answers to score trials against. Pick an evaluation run from Studio.'
    );
  }
  const configFile = await findEvalConfigFile(workspace, filesetName);
  if (!configFile) {
    throw new Error(`Could not find the eval config for evaluation "${evaluation.name}".`);
  }

  const rows = await studyRowsFromSpec(
    parseEvalConfig(await downloadText(workspace, filesetName, configFile)),
    downloadText
  );
  if (rows.length === 0) throw new Error(`Evaluation "${evaluation.name}" has no rows.`);
  return rows;
};
