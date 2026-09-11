// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import {
  evaluationFilesetName,
  findEvalConfigFile,
} from '@studio/components/evaluation/experimentEvalConfig';
import {
  isDatasetEvalSpec,
  type EvalSpec,
  type EvalSpecTask,
} from '@studio/components/evaluation/submitEvaluationJob';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { parse as parseYaml } from 'yaml';

/** Flat filename the study's rows are staged under, beside the generated config. The job chdirs to
 *  the staged bundle root before reading, so a bare filename is what `eval.general.dataset`
 *  resolves against. */
export const STUDY_DATASET_PATH = 'dataset.json';

/**
 * One row as the optimize trial path reads it.
 *
 * `instruction` is the prompt handed to the agent and `answer` is what the judge scores against —
 * the two fields `build_agent_eval_tasks` looks for. Everything else in the source config is
 * dropped: the study re-runs the agent itself, so a stored result would only be stale.
 */
export interface StudyDatasetRow {
  id: string;
  instruction: string;
  answer: string;
}

export class StudyDatasetError extends Error {}

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;

/** First non-empty string among `keys`, which is how the row shorthands are collapsed. */
const firstText = (row: Record<string, unknown>, keys: string[]): string | undefined => {
  for (const key of keys) {
    const value = row[key];
    if (typeof value === 'string' && value.trim()) return value;
    if (typeof value === 'number') return String(value);
  }
  return undefined;
};

const taskToRow = (task: EvalSpecTask, index: number): StudyDatasetRow => {
  const instruction = task.inputs?.instruction?.trim() || task.intent?.trim();
  if (!instruction) {
    throw new StudyDatasetError(
      `Task "${task.id || index}" has no instruction, so there is nothing to prompt the agent with.`
    );
  }
  const reference = asRecord(task.reference) ?? {};
  return {
    id: String(task.id || index),
    instruction,
    answer: firstText(reference, ['answer', 'expected_answer', 'reference', 'label']) ?? '',
  };
};

const datasetRowToRow = (raw: unknown, index: number): StudyDatasetRow => {
  const row = asRecord(raw);
  const instruction = row && firstText(row, ['instruction', 'question', 'prompt', 'input', 'body']);
  if (!row || !instruction) {
    throw new StudyDatasetError(
      `Row ${index} has no instruction/question/prompt, so there is nothing to prompt the agent with.`
    );
  }
  return {
    id: firstText(row, ['id']) ?? String(index),
    instruction,
    answer: firstText(row, ['answer', 'expected_answer', 'reference', 'label']) ?? '',
  };
};

/** Rows an eval config carries, whichever of the two stored shapes it is. */
export const studyRowsFromEvalSpec = (spec: EvalSpec): StudyDatasetRow[] => {
  if (isDatasetEvalSpec(spec)) {
    if (!Array.isArray(spec.dataset)) {
      throw new StudyDatasetError(
        'This evaluation names its dataset by reference rather than storing rows, which the study cannot read. Pick another evaluation.'
      );
    }
    return spec.dataset.map(datasetRowToRow);
  }
  const tasks = Array.isArray(spec.tasks) ? spec.tasks : [];
  if (tasks.length === 0) {
    throw new StudyDatasetError('This evaluation has no tasks to score trials against.');
  }
  return tasks.map(taskToRow);
};

const parseEvalConfig = (path: string, text: string): EvalSpec => {
  const parsed = path.endsWith('.json') ? JSON.parse(text) : parseYaml(text);
  if (!asRecord(parsed)) {
    throw new StudyDatasetError(`Eval config "${path}" is not an object.`);
  }
  return parsed as EvalSpec;
};

/**
 * The rows a study will replay, read out of the evaluation the user picked.
 *
 * The study runs the agent itself rather than reusing the evaluation's recorded outputs — the whole
 * point is to score a *changed* config — so what it needs from the evaluation is its prompts and
 * expected answers. Those live in the eval config the run was submitted with, which Studio stores
 * in the evaluation's own fileset.
 */
export const loadStudyDataset = async (
  workspace: string,
  evaluation: AgentEvaluationRow,
  signal?: AbortSignal
): Promise<StudyDatasetRow[]> => {
  const fileset = evaluationFilesetName(evaluation) ?? evaluation.dataset_name;
  if (!fileset) {
    throw new StudyDatasetError(
      `Evaluation "${evaluation.name}" names no fileset, so its dataset cannot be read. Pick another evaluation.`
    );
  }

  const configPath = await findEvalConfigFile(workspace, fileset, signal);
  if (!configPath) {
    throw new StudyDatasetError(
      `Fileset "${fileset}" has no eval config at its root, so evaluation "${evaluation.name}" cannot be reused. Pick another evaluation.`
    );
  }

  const blob = await filesDownloadFile(workspace, fileset, configPath, signal);
  const rows = studyRowsFromEvalSpec(parseEvalConfig(configPath, await blob.text()));
  if (rows.length === 0) {
    throw new StudyDatasetError('This evaluation has no rows to score trials against.');
  }
  return rows;
};
