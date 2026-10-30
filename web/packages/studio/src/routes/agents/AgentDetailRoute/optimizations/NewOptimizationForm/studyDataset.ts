// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDownloadFile, filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import { readParquetRows } from '@studio/api/datasets/filesetParquetRows';
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
const BINARY_DATASET = /\.(feather|arrow|orc)$/i;
const PARQUET_DATASET = /\.parquet$/i;
const GZIP_DATASET = /\.(gz|gzip)$/i;

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

const gunzip = async (blob: Blob, path: string): Promise<Blob> => {
  try {
    return await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).blob();
  } catch {
    throw new Error(`Could not decompress ${path} as gzip.`);
  }
};

/** Decode a downloaded dataset the way the evaluator's loader does: gzip wraps any format. */
export const readDatasetRecords = async (
  blob: Blob,
  path: string
): Promise<Record<string, unknown>[]> => {
  if (GZIP_DATASET.test(path))
    return readDatasetRecords(await gunzip(blob, path), path.replace(GZIP_DATASET, ''));
  if (PARQUET_DATASET.test(path)) return readParquetRows(blob);
  return parseDatasetRecords(await blob.text(), path);
};

const GLOB_SEGMENT = /[*?[\]]/;

/** Python's `Path.glob` over fileset paths: `*` and `?` stay within a segment, `**` spans them. */
export const globToRegExp = (pattern: string): RegExp => {
  let source = '';
  for (let index = 0; index < pattern.length; index += 1) {
    const char = pattern[index];
    if (pattern.startsWith('**/', index)) {
      source += '(?:[^/]*/)*';
      index += 2;
    } else if (pattern.startsWith('**', index)) {
      source += '.*';
      index += 1;
    } else if (char === '*') {
      source += '[^/]*';
    } else if (char === '?') {
      source += '[^/]';
    } else if (char === '[' && pattern.indexOf(']', index + 2) > 0) {
      const end = pattern.indexOf(']', index + 2);
      const body = pattern.slice(index + 1, end).replace(/\\/g, '\\\\');
      source += `[${body.startsWith('!') ? `^${body.slice(1)}` : body}]`;
      index = end;
    } else {
      source += char.replace(/[.+^${}()|[\]\\/]/g, '\\$&');
    }
  }
  return new RegExp(`^${source}$`);
};

const globPrefix = (pattern: string): string | undefined => {
  const segments = pattern.split('/');
  const literal = segments.slice(
    0,
    segments.findIndex((segment) => GLOB_SEGMENT.test(segment))
  );
  return literal.length > 0 ? `${literal.join('/')}/` : undefined;
};

export interface DatasetSource {
  listFiles: (workspace: string, fileset: string, prefix?: string) => Promise<string[]>;
  download: (workspace: string, fileset: string, path: string) => Promise<Blob>;
}

/** A glob ref concatenates its matches in sorted order, as the evaluator's loader does. */
const readDatasetRef = async (
  source: DatasetSource,
  workspace: string,
  fileset: string,
  path: string
): Promise<Record<string, unknown>[]> => {
  if (!GLOB_SEGMENT.test(path))
    return readDatasetRecords(await source.download(workspace, fileset, path), path);
  const matcher = globToRegExp(path);
  const matches = (await source.listFiles(workspace, fileset, globPrefix(path)))
    .filter((file) => matcher.test(file))
    .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  if (matches.length === 0)
    throw new Error(`No files in fileset "${fileset}" match the evaluation's dataset ${path}.`);
  const parts = await Promise.all(
    matches.map(async (file) =>
      readDatasetRecords(await source.download(workspace, fileset, file), file)
    )
  );
  return parts.flat();
};

export const studyEvaluationFromSpec = async (
  spec: EvalSpec,
  source: DatasetSource
): Promise<StudyEvaluation> => {
  if (!isDatasetEvalSpec(spec)) return { spec, records: spec.tasks.map((task) => ({ ...task })) };
  if (Array.isArray(spec.dataset)) return { spec, records: spec.dataset };
  const ref = DATASET_REF.exec(spec.dataset);
  if (!ref)
    throw new Error(
      `The evaluation's dataset "${spec.dataset}" is not a workspace/fileset#path ref.`
    );
  const [, workspace, fileset, path] = ref;
  return { spec, records: await readDatasetRef(source, workspace, fileset, path) };
};

const downloadFile = async (workspace: string, fileset: string, path: string): Promise<Blob> => {
  const blob = await filesDownloadFile(workspace, fileset, path);
  if (!blob) throw new Error(`Could not read ${path} from fileset "${fileset}".`);
  return blob;
};

const filesetSource: DatasetSource = {
  listFiles: async (workspace, fileset, prefix) =>
    (
      await filesListFilesetFiles(workspace, fileset, prefix ? { path: prefix } : undefined)
    ).data.map((file) => file.path),
  download: downloadFile,
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
    parseEvalConfig(await (await downloadFile(workspace, filesetName, configFile)).text()),
    filesetSource
  );
  if (evaluationData.records.length === 0)
    throw new Error(`Evaluation "${evaluation.name}" has no rows.`);
  return evaluationData;
};
