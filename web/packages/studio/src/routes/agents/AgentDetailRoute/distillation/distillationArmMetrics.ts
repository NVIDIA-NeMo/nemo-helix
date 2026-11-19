// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getEvaluation } from '@nemo/sdk/generated/platform/evaluations';
import type { EvaluatorAggregate } from '@nemo/sdk/generated/platform/schema';
import { getTraceMetrics } from '@nemo/sdk/generated/platform/traces';
import type { DistillationWorkflow } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import type { ArmId } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';

export interface ArmMetrics {
  score: number | null;
  inputTokensPerTask: number | null;
  outputTokensPerTask: number | null;
  medianLatencySec: number | null;
  costPer1kTasks: number | null;
}

const finite = (value: number | null | undefined): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/** The headline metric: a reward output when there is one, else the first evaluator with a mean. */
const headlineScore = (scores: Record<string, EvaluatorAggregate> | undefined): number | null => {
  const entries = Object.entries(scores ?? {}).filter(([, aggregate]) => finite(aggregate.mean) !== null);
  const chosen = entries.find(([name]) => name.endsWith('.reward')) ?? entries[0];
  return chosen ? finite(chosen[1].mean) : null;
};

/**
 * Each arm's Intake Evaluation, keyed by arm. The distilled arm is the candidate's: the one key that
 * is neither the original nor the base arm.
 */
export const evaluationNamesByArm = (workflow: DistillationWorkflow): Partial<Record<ArmId, string>> => {
  const evaluations = workflow.nodes.experiment?.outputs.evaluations;
  if (typeof evaluations !== 'object' || evaluations === null) return {};
  const names = Object.entries(evaluations as Record<string, unknown>).filter(
    (entry): entry is [string, string] => typeof entry[1] === 'string'
  );
  const find = (arm: string) => names.find(([key]) => key === arm)?.[1];
  const distilled = names.find(([key]) => key !== 'original' && key !== 'base')?.[1];
  return { original: find('original'), base: find('base'), distilled };
};

/**
 * Score, tokens, latency and cost for one evaluation, read from Intake. The trace rollup is bounded to
 * runs since the workflow was created, because without a lower bound it covers only the last 7 days.
 */
export const fetchArmMetrics = async (
  workspace: string,
  evaluationName: string,
  since: string | undefined
): Promise<ArmMetrics> => {
  const [evaluation, metrics] = await Promise.all([
    getEvaluation(workspace, evaluationName),
    getTraceMetrics(workspace, {
      bucket: 'total',
      filter: {
        evaluation_name: evaluationName,
        ...(since ? { started_at: { $gte: since } } : {}),
      },
    }),
  ]);
  const point = metrics.data[0];
  const latencyMs = finite(point?.latency_ms.p50);
  const cost = finite(point?.cost_usd.mean);
  return {
    score: headlineScore(evaluation.aggregate_scores),
    inputTokensPerTask: finite(point?.input_tokens.mean),
    outputTokensPerTask: finite(point?.output_tokens.mean),
    medianLatencySec: latencyMs === null ? null : latencyMs / 1000,
    costPer1kTasks: cost === null ? null : cost * 1000,
  };
};
