// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import { filesDownloadFile, filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import type { ArmMetrics } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationArmMetrics';

const SUMMARY_FILE = 'summary.json';
const TRIALS_FILE = 'trials.jsonl';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const finite = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const mean = (values: number[]): number | null =>
  values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;

const median = (values: number[]): number | null => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

/** The headline mean of a run summary: a reward output when there is one, else the first plain metric. */
const scoreOf = (summary: unknown): number | null => {
  const scores = isRecord(summary) && isRecord(summary.scores) ? summary.scores.scores : undefined;
  if (!Array.isArray(scores)) return null;
  const plain = scores.filter(
    (entry): entry is Record<string, unknown> =>
      isRecord(entry) &&
      typeof entry.name === 'string' &&
      !entry.name.startsWith('view.') &&
      !entry.name.includes('.pass@') &&
      finite(entry.mean) !== null
  );
  const chosen = plain.find((entry) => String(entry.name).endsWith('.reward')) ?? plain[0];
  return chosen ? finite(chosen.mean) : null;
};

const pathEndingWith = (paths: string[], file: string): string | undefined =>
  paths.filter((path) => path === file || path.endsWith(`/${file}`)).sort((a, b) => a.length - b.length)[0];

interface TrialFigures {
  input: number[];
  output: number[];
  seconds: number[];
}

const trialFigures = (lines: string): TrialFigures => {
  const figures: TrialFigures = { input: [], output: [], seconds: [] };
  lines
    .split('\n')
    .filter((line) => line.trim())
    .forEach((line) => {
      try {
        const trial: unknown = JSON.parse(line);
        const measurements = isRecord(trial) && isRecord(trial.measurements) ? trial.measurements : undefined;
        const input = finite(measurements?.prompt_tokens);
        const output = finite(measurements?.completion_tokens);
        const seconds = finite(measurements?.runtime_sec);
        if (input !== null) figures.input.push(input);
        if (output !== null) figures.output.push(output);
        if (seconds !== null) figures.seconds.push(seconds);
      } catch {
        // A line that is not a trial is skipped.
      }
    });
  return figures;
};

/** The model the agent was registered with, for pricing the baseline. */
const agentModel = async (workspace: string, agent: string): Promise<string | undefined> => {
  const [agentWorkspace, name] = agent.includes('/') ? agent.split('/', 2) : [workspace, agent];
  const config = (await agentsGetAgent(agentWorkspace, name)).config;
  const models = isRecord(config) && isRecord(config.models) ? Object.values(config.models) : [];
  const first = models.find(isRecord);
  return typeof first?.model === 'string' ? first.model : undefined;
};

/**
 * The baseline arm, read from the evaluation run's own results fileset: the score from its summary, and
 * tokens and runtime from the trials the run recorded. Cost is left for the caller to price.
 */
export const fetchBaselineMetrics = async (
  workspace: string,
  filesetRef: string,
  sourceAgent: string
): Promise<ArmMetrics> => {
  const [filesetWorkspace, fileset] = filesetRef.includes('/') ? filesetRef.split('/', 2) : [workspace, filesetRef];
  const listing = await filesListFilesetFiles(filesetWorkspace, fileset);
  const paths = (listing.data ?? []).map((file) => file.path);
  const summaryPath = pathEndingWith(paths, SUMMARY_FILE);
  const trialsPath = pathEndingWith(paths, TRIALS_FILE);

  const [summaryText, trialsText, model] = await Promise.all([
    summaryPath ? filesDownloadFile(filesetWorkspace, fileset, summaryPath).then((blob) => blob.text()) : null,
    trialsPath ? filesDownloadFile(filesetWorkspace, fileset, trialsPath).then((blob) => blob.text()) : null,
    agentModel(workspace, sourceAgent).catch(() => undefined),
  ]);
  const figures = trialsText ? trialFigures(trialsText) : { input: [], output: [], seconds: [] };
  let summary: unknown = null;
  try {
    summary = summaryText ? JSON.parse(summaryText) : null;
  } catch {
    summary = null;
  }
  return {
    score: scoreOf(summary),
    inputTokensPerTask: mean(figures.input),
    outputTokensPerTask: mean(figures.output),
    medianLatencySec: median(figures.seconds),
    costPer1kTasks: null,
    model,
  };
};
