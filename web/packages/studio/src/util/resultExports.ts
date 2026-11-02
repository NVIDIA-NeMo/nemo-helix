// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CsvRow, DownloadPayload } from '@nemo/common/src/utils/download';
import type { EvaluationSessionResponse } from '@nemo/sdk/generated/platform/schema';
import type { AgentEvalTaskDetail } from '@studio/api/evaluation/agent-evaluations';
import type { EvalAggregateScoreRow } from '@studio/components/evaluation/EvalAggregateScoresTable';
import type { DatasetEvalRow } from '@studio/components/evaluation/Jobs/datasetEval/DatasetEvalRowResultsPanel';
import {
  expectedValue,
  inputText,
  scoreCells,
} from '@studio/components/evaluation/Jobs/datasetEval/rowFields';
import type { CustomizationMetricSeries } from '@studio/types/customization';

/** JSONL and CSV hold the per-row records; a run with no rows falls back to its aggregate scores. */
const rowsOrScores = <T>(rows: T[], scores: readonly EvalAggregateScoreRow[]): (T | CsvRow)[] =>
  rows.length > 0 ? rows : scores.map((score) => ({ ...score }));

export const datasetEvalExport = (
  job: { name?: string; status?: string },
  scores: EvalAggregateScoreRow[],
  rows: DatasetEvalRow[]
): DownloadPayload => ({
  json: { name: job.name, status: job.status, scores, rows },
  jsonl: rowsOrScores(rows, scores),
  csv: rowsOrScores(
    rows.map((row, index) => ({
      row: row.row_index ?? index,
      input: inputText(row),
      expected: expectedValue(row.item),
      response: row.sample?.output_text,
      ...Object.fromEntries(scoreCells(row).map(({ label, value }) => [label, value])),
    })),
    scores
  ),
});

export const agentEvalExport = (
  job: { name: string; status?: string; agent: string | null },
  scores: EvalAggregateScoreRow[],
  tasks: AgentEvalTaskDetail[],
  summary?: unknown
): DownloadPayload => ({
  json: { ...job, scores, summary, tasks },
  jsonl: rowsOrScores(tasks, scores),
  csv: rowsOrScores(
    tasks.map((task) => ({
      task_id: task.taskId,
      intent: task.intent,
      instruction: task.instruction,
      reference: task.reference,
      status: task.status,
      response: task.responseText,
      ...Object.fromEntries(task.scores.map(({ name, value }) => [name, value])),
      diagnostics: task.diagnostics.length > 0 ? task.diagnostics : undefined,
    })),
    scores
  ),
});

export const evaluationSessionsExport = (
  evaluation: unknown,
  sessions: EvaluationSessionResponse[]
): DownloadPayload => ({
  json: { evaluation, sessions },
  jsonl: sessions,
  csv: sessions.map(({ evaluator_scores, failed_evaluators, ...session }) => ({
    ...session,
    ...evaluator_scores,
    ...Object.fromEntries((failed_evaluators ?? []).map((name) => [name, 'FAILED'])),
  })),
});

/** One row per training step, with a field per reported metric series. */
export const customizationMetricsExport = (
  job: { name?: string; status?: string },
  metrics: CustomizationMetricSeries = {}
): DownloadPayload => {
  const byStep = new Map<number, CsvRow>();
  for (const [name, series] of Object.entries(metrics)) {
    for (const { step, epoch, value } of series ?? []) {
      const row = byStep.get(step) ?? { step };
      row.epoch ??= epoch;
      row[name] = value;
      byStep.set(step, row);
    }
  }
  const steps = [...byStep.values()].sort((a, b) => Number(a.step) - Number(b.step));
  return { json: { ...job, metrics }, jsonl: steps, csv: steps };
};
