// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { agentsListOptimizeJobResults } from '@nemo/sdk/generated/agents/agents';
import { filesDownloadFile, filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { FileStorageType } from '@nemo/sdk/generated/platform/schema';
import Papa from 'papaparse';

const SUMMARY_FILE = 'study_summary.json';
const TRIALS_FILE = 'trials_dataframe_params.csv';

/** One row of {@link TRIALS_FILE}, before the `values_`/`params_` columns are split out. */
type TrialCsvRow = Record<string, string | undefined>;

export interface StudySummary {
  nTrials: number | null;
  bestTrial: number | null;
  metricNames: string[];
  bestValues: number[];
}

export interface TrialMetric {
  name: string;
  value: number | null;
}

export interface TrialParam {
  name: string;
  value: string;
}

export interface Trial {
  number: number;
  state: string;
  durationSeconds: number | null;
  paretoOptimal: boolean;
  metrics: TrialMetric[];
  params: TrialParam[];
}

export interface StudyResults {
  summary: StudySummary | null;
  trials: Trial[];
  metricNames: string[];
}

const toNumber = (value: string | number | null | undefined): number | null => {
  if (value === null || value === undefined || value === '') return null;
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

/**
 * Parse `str(timedelta)` — `H:MM:SS[.ffffff]`, with an optional leading `N day(s), `.
 * Returns null for the empty string the writer emits for a trial that never completed.
 */
export const parseDurationSeconds = (value: string | undefined): number | null => {
  if (!value) return null;
  const match = /^(?:(\d+)\s+days?,\s*)?(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$/.exec(value.trim());
  if (!match) return null;
  const [, days, hours, minutes, seconds] = match;
  return (
    Number(days ?? 0) * 86_400 + Number(hours) * 3_600 + Number(minutes) * 60 + Number(seconds)
  );
};

/** Python's `csv` writer emits bare `True`/`False` for booleans. */
const parseBoolean = (value: string | undefined): boolean => (value ?? '').toLowerCase() === 'true';

const parseSummary = (text: string): StudySummary => {
  const raw = JSON.parse(text) as Record<string, unknown>;
  const metricNames = Array.isArray(raw.metric_names) ? raw.metric_names.map(String) : [];
  const bestValues = Array.isArray(raw.best_values)
    ? raw.best_values.map((v) => toNumber(v as number)).filter((v): v is number => v !== null)
    : [];
  return {
    nTrials: toNumber(raw.n_trials as number),
    bestTrial: toNumber(raw.best_trial as number),
    metricNames,
    bestValues,
  };
};

/**
 * Split each CSV row into its fixed columns, its `values_<metric>` metrics and its
 * `params_<name>` parameters. Metric order follows `metricNames` when the summary supplied it,
 * so the table's columns match the objective order the study was configured with; otherwise it
 * falls back to header order.
 */
const parseTrials = (
  text: string,
  metricNames: string[]
): { trials: Trial[]; metrics: string[] } => {
  const parsed = Papa.parse<TrialCsvRow>(text, { header: true, skipEmptyLines: true });
  const headers = parsed.meta.fields ?? [];

  const headerMetrics = headers
    .filter((h) => h.startsWith('values_'))
    .map((h) => h.slice('values_'.length));
  const metrics = metricNames.length
    ? metricNames.filter((name) => headerMetrics.includes(name))
    : headerMetrics;
  const paramNames = headers
    .filter((h) => h.startsWith('params_'))
    .map((h) => h.slice('params_'.length));

  const trials = parsed.data.flatMap<Trial>((row) => {
    const number = toNumber(row.number);
    if (number === null) return [];
    return [
      {
        number,
        state: row.state ?? '',
        durationSeconds: parseDurationSeconds(row.duration),
        paretoOptimal: parseBoolean(row.pareto_optimal),
        metrics: metrics.map((name) => ({ name, value: toNumber(row[`values_${name}`]) })),
        params: paramNames
          .map((name) => ({ name, value: row[`params_${name}`] ?? '' }))
          .filter((param) => param.value !== ''),
      },
    ];
  });

  return { trials, metrics };
};

/**
 * Locate the study artifacts inside the job's registered results.
 */
const locateStudyFiles = async (
  workspace: string,
  jobName: string,
  signal?: AbortSignal
): Promise<{ fileset: string; summaryPath?: string; trialsPath?: string } | null> => {
  const { data: results } = await agentsListOptimizeJobResults(workspace, jobName, signal);

  for (const result of results) {
    if (result.artifact_storage_type !== FileStorageType.fileset) continue;
    const parsed = parseFilesetLocation(result.artifact_url, workspace);
    if (!parsed) continue;

    const { data: files } = await filesListFilesetFiles(
      parsed.workspace,
      parsed.name,
      { path: parsed.objectPath || undefined },
      signal
    );
    const at = (fileName: string) =>
      files.find((file) => file.path === fileName || file.path.endsWith(`/${fileName}`))?.path;

    const summaryPath = at(SUMMARY_FILE);
    const trialsPath = at(TRIALS_FILE);
    if (summaryPath ?? trialsPath) {
      return { fileset: parsed.name, summaryPath, trialsPath };
    }
  }

  return null;
};

const downloadText = async (
  workspace: string,
  fileset: string,
  path: string,
  signal?: AbortSignal
): Promise<string | null> => {
  const blob = await filesDownloadFile(workspace, fileset, path, signal);
  return blob ? blob.text() : null;
};

/**
 * Read the study summary and the per-trial table for one optimize job.
 */
export const fetchStudyResults = async (
  workspace: string,
  jobName: string,
  signal?: AbortSignal
): Promise<StudyResults | null> => {
  const located = await locateStudyFiles(workspace, jobName, signal);
  if (!located) return null;

  const { fileset, summaryPath, trialsPath } = located;
  const [summaryText, trialsText] = await Promise.all([
    summaryPath ? downloadText(workspace, fileset, summaryPath, signal) : null,
    trialsPath ? downloadText(workspace, fileset, trialsPath, signal) : null,
  ]);

  const summary = summaryText ? parseSummary(summaryText) : null;
  const { trials, metrics } = trialsText
    ? parseTrials(trialsText, summary?.metricNames ?? [])
    : { trials: [], metrics: summary?.metricNames ?? [] };

  return { summary, trials, metricNames: metrics };
};
