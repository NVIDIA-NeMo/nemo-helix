// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { agentOptimizationListRunStrategyJobResults } from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import { filesDownloadFile, filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { FileStorageType } from '@nemo/sdk/generated/platform/schema';
import Papa from 'papaparse';

const SUMMARY_FILE = 'study_summary.json';
const TRIALS_FILE = 'trials_dataframe_params.csv';
const PROMPT_SUMMARY_FILE = 'prompt_phase_summary.json';
const PROMPT_HISTORY_FILE = 'ga_history_prompts.json';

/** Column prefixes of a {@link PROMPT_HISTORY_FILE} row. */
const PROMPT_COLUMN = 'prompt.';
const METRIC_COLUMN = 'metric.';

/** One row of {@link TRIALS_FILE}, before the `values_`/`params_` columns are split out. */
type TrialCsvRow = Record<string, string | undefined>;

export interface StudySummary {
  nTrials: number | null;
  bestTrial: number | null;
  metricNames: string[];
  bestValues: (number | null)[];
}

export interface TrialMetric {
  name: string;
  value: number | null;
}

export interface TrialParam {
  name: string;
  value: string;
}

/** `numeric` trials come from the parameter sweep; `prompt` trials are the GA's rewritten
 *  prompts, which it scores from the sweep's best parameters. */
export type TrialPhase = 'numeric' | 'prompt';

export interface Trial {
  number: number;
  phase: TrialPhase;
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
    ? raw.best_values.map((v) => toNumber(v as number))
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
        phase: 'numeric',
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

interface PromptPhase {
  bestTrial: number | null;
  bestMetrics: Record<string, unknown>;
}

const parsePromptSummary = (text: string): PromptPhase => {
  const raw = JSON.parse(text) as Record<string, unknown>;
  const bestMetrics = raw.best_metrics;
  return {
    bestTrial: toNumber(raw.best_global_trial_number as number),
    bestMetrics:
      typeof bestMetrics === 'object' && bestMetrics !== null
        ? (bestMetrics as Record<string, unknown>)
        : {},
  };
};

/** The GA's lowercase individual statuses, in Optuna's `TrialState` vocabulary the table uses. */
const PROMPT_STATES: Record<string, string> = { completed: 'COMPLETE', failed: 'FAIL' };

/**
 * One trial per GA individual, numbered as the study numbers it. Each ran its prompts on the
 * sweep's best parameters, so those join its params and a deploy applies both.
 */
const parsePromptTrials = (
  text: string,
  metricNames: string[],
  baseParams: TrialParam[]
): Trial[] => {
  const rows: unknown = JSON.parse(text);
  if (!Array.isArray(rows)) return [];
  return rows.flatMap<Trial>((row: Record<string, unknown>) => {
    const number = toNumber(row.global_trial_number as number);
    if (number === null) return [];
    const prompts = Object.entries(row)
      .filter(
        (entry): entry is [string, string] =>
          entry[0].startsWith(PROMPT_COLUMN) && typeof entry[1] === 'string'
      )
      .map(([key, value]) => ({ name: key.slice(PROMPT_COLUMN.length), value }));
    const status = String(row.status ?? '');
    return [
      {
        number,
        phase: 'prompt',
        state: PROMPT_STATES[status] ?? status.toUpperCase(),
        durationSeconds: null,
        paretoOptimal: false,
        metrics: metricNames.map((name) => ({
          name,
          value: toNumber(row[`${METRIC_COLUMN}${name}`] as number),
        })),
        params: [
          ...baseParams.filter((param) => !prompts.some((prompt) => prompt.name === param.name)),
          ...prompts,
        ],
      },
    ];
  });
};

export interface JobResultFiles {
  workspace: string;
  fileset: string;
  /** Fileset path of each requested file name the result holds; absent when it holds none. */
  paths: Partial<Record<string, string>>;
}

/**
 * Locate the named files inside an optimize job's registered results: the first fileset-backed
 * result holding any of them wins.
 */
export const locateJobResultFiles = async (
  workspace: string,
  jobName: string,
  fileNames: readonly string[],
  signal?: AbortSignal
): Promise<JobResultFiles | null> => {
  const { data: results } = await agentOptimizationListRunStrategyJobResults(
    workspace,
    jobName,
    undefined,
    signal
  );

  for (const result of results) {
    if (result.artifact_storage_type !== FileStorageType.fileset) continue;
    const parsed = parseFilesetLocation(result.artifact_url, workspace);
    if (!parsed) continue;

    const { data: files } = await filesListFilesetFiles(
      parsed.workspace,
      parsed.name,
      { path: parsed.filesListPathPrefix },
      signal
    );
    const paths: Partial<Record<string, string>> = {};
    for (const fileName of fileNames) {
      const path = files.find(
        (file) => file.path === fileName || file.path.endsWith(`/${fileName}`)
      )?.path;
      if (path) paths[fileName] = path;
    }
    if (Object.keys(paths).length) {
      return { workspace: parsed.workspace, fileset: parsed.name, paths };
    }
  }

  return null;
};

export const downloadText = async (
  workspace: string,
  fileset: string,
  path: string,
  signal?: AbortSignal
): Promise<string | null> => {
  const blob = await filesDownloadFile(workspace, fileset, path, signal);
  return blob ? blob.text() : null;
};

/**
 * Read the study summary and the per-trial table for one optimize job. When the study also ran
 * the prompt phase, its candidates follow the sweep's trials and its winner, which the study
 * ends on, replaces the sweep's best.
 */
export const fetchStudyResults = async (
  workspace: string,
  jobName: string,
  signal?: AbortSignal
): Promise<StudyResults | null> => {
  const located = await locateJobResultFiles(
    workspace,
    jobName,
    [SUMMARY_FILE, TRIALS_FILE, PROMPT_SUMMARY_FILE, PROMPT_HISTORY_FILE],
    signal
  );
  if (!located) return null;

  const { workspace: artifactWorkspace, fileset, paths } = located;
  const download = (name: string) => {
    const path = paths[name];
    return path ? downloadText(artifactWorkspace, fileset, path, signal) : null;
  };
  const [summaryText, trialsText, promptSummaryText, promptHistoryText] = await Promise.all([
    download(SUMMARY_FILE),
    download(TRIALS_FILE),
    download(PROMPT_SUMMARY_FILE),
    download(PROMPT_HISTORY_FILE),
  ]);

  const summary = summaryText ? parseSummary(summaryText) : null;
  const { trials, metrics } = trialsText
    ? parseTrials(trialsText, summary?.metricNames ?? [])
    : { trials: [], metrics: summary?.metricNames ?? [] };
  if (!promptHistoryText) return { summary, trials, metricNames: metrics };

  const sweepBest = trials.find((trial) => trial.number === summary?.bestTrial);
  const promptTrials = parsePromptTrials(promptHistoryText, metrics, sweepBest?.params ?? []);
  const promptPhase = promptSummaryText ? parsePromptSummary(promptSummaryText) : null;
  const promptBest = promptTrials.find((trial) => trial.number === promptPhase?.bestTrial);
  const bestTrial = promptBest?.number ?? summary?.bestTrial ?? null;
  const allTrials = [...trials, ...promptTrials].map((trial) => ({
    ...trial,
    paretoOptimal: trial.paretoOptimal || trial.number === bestTrial,
  }));

  return {
    summary: {
      nTrials: allTrials.length,
      bestTrial,
      metricNames: metrics,
      bestValues: promptBest
        ? metrics.map(
            (name, i) =>
              toNumber(promptPhase?.bestMetrics[name] as number) ?? promptBest.metrics[i].value
          )
        : (summary?.bestValues ?? []),
    },
    trials: allTrials,
    metricNames: metrics,
  };
};
