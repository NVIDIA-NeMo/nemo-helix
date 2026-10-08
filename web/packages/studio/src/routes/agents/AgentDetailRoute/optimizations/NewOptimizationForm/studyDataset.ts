// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import {
  evaluationFilesetName,
  findEvalConfigFile,
} from '@studio/components/evaluation/experimentEvalConfig';
import {
  type EvalSpec,
  isDatasetEvalSpec,
  parseEvalConfig,
} from '@studio/components/evaluation/submitEvaluationJob';
import { asRecord } from '@studio/util/guards';
import Papa from 'papaparse';

/** Keep the authored scoring context intact; answers are required only by metrics that use them. */
export interface StudyEvaluation {
  spec: EvalSpec;
  records: Record<string, unknown>[];
}

const DATASET_REF = /^([\w\-.]+)\/([\w\-.]+)#(.+)$/;
const BINARY_DATASET = /\.(parquet|feather|arrow|orc|gz|gzip)$/i;

/** Read the evaluator's text dataset formats without discarding the judge's row context. */
export const parseDatasetRecords = (text: string, path = ''): Record<string, unknown>[] => {
  if (/\.csv$/i.test(path)) {
    const { data, errors } = Papa.parse<Record<string, string>>(text, {
      header: true,
      skipEmptyLines: true,
    });
    const [error] = errors;
    if (error)
      throw new Error(`Could not parse ${path} as CSV: ${error.message} (row ${error.row}).`);
    return data;
  }
  if (BINARY_DATASET.test(path)) {
    throw new Error(
      `The evaluation's dataset ${path} is not JSON, JSON Lines or CSV, so the study cannot stage it.`
    );
  }
  let records: unknown[];
  try {
    const parsed: unknown = JSON.parse(text.trim());
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

export const studyEvaluationFromSpec = async (
  spec: EvalSpec,
  readDataset: (workspace: string, fileset: string, path: string) => Promise<string>
): Promise<StudyEvaluation> => {
  if (!isDatasetEvalSpec(spec)) return { spec, records: spec.tasks.map((task) => ({ ...task })) };
  if (Array.isArray(spec.dataset)) return { spec, records: spec.dataset };
  const ref = DATASET_REF.exec(spec.dataset);
  if (!ref)
    throw new Error(
      `The evaluation's dataset "${spec.dataset}" is not a workspace/fileset#path ref.`
    );
  const [, workspace, fileset, path] = ref;
  return { spec, records: parseDatasetRecords(await readDataset(workspace, fileset, path), path) };
};

const downloadText = async (workspace: string, fileset: string, path: string): Promise<string> => {
  const blob = await filesDownloadFile(workspace, fileset, path);
  if (!blob) throw new Error(`Could not read ${path} from fileset "${fileset}".`);
  return blob.text();
};

/** Intake omits grading context; use the exact config saved on the selected evaluation. */
export const loadStudyEvaluation = async (
  workspace: string,
  evaluation: EvaluationResponse
): Promise<StudyEvaluation> => {
  const filesetName = evaluationFilesetName(evaluation);
  if (!filesetName) {
    throw new Error(
      `Evaluation "${evaluation.name}" has no stored eval config. Pick an evaluation run from Studio.`
    );
  }
  const configFile = await findEvalConfigFile(workspace, filesetName);
  if (!configFile)
    throw new Error(`Could not find the eval config for evaluation "${evaluation.name}".`);
  const evaluationData = await studyEvaluationFromSpec(
    parseEvalConfig(await downloadText(workspace, filesetName, configFile)),
    downloadText
  );
  if (evaluationData.records.length === 0)
    throw new Error(`Evaluation "${evaluation.name}" has no rows.`);
  return evaluationData;
};
