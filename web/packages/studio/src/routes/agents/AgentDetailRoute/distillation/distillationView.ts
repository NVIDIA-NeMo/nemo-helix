// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DistillationWorkflow,
  WorkflowArm,
  WorkflowStage,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import type { ArmMetrics } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationArmMetrics';
import type {
  ArmId,
  MockArmResult,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import type { MockStage } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationWorkflow';
import {
  estimateCostPer1kTasks,
  listedPrice,
} from '@studio/routes/agents/AgentDetailRoute/distillation/modelPricing';

const STAGE_DESCRIPTIONS: Record<string, string> = {
  prepare: 'Turn passing trials from the original evaluations into training rows.',
  customization: 'Train the student model on the prepared rows.',
  'eval-original': 'Baseline: the agent as it runs today, on the held-out evaluations.',
  'eval-base': 'The same agent with the student model and no training.',
  'eval-distilled': 'The same agent with the trained student model.',
};

const humanize = (key: string): string => {
  const spaced = key.replace(/_/g, ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
};

const formatMetric = (value: unknown): string => {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
};

export const toStageView = (stage: WorkflowStage): MockStage => ({
  id: stage.id as MockStage['id'],
  title: stage.name,
  description: STAGE_DESCRIPTIONS[stage.id] ?? '',
  status: stage.status,
  progress: stage.progress,
  jobRef: stage.job_ref ? { kind: 'Job', name: stage.job_ref } : undefined,
  metrics: Object.entries(stage.metrics)
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([key, value]) => ({ label: humanize(key), value: formatMetric(value) })),
});

const toArm = (
  id: MockArmResult['id'],
  title: string,
  model: string,
  arm: WorkflowArm | null,
  fallback: ArmMetrics | undefined,
  priceModel: string | undefined
): MockArmResult | null => {
  const score = arm?.score ?? fallback?.score ?? null;
  const inputTokens = arm?.input_tokens_per_task ?? fallback?.inputTokensPerTask ?? null;
  const outputTokens = arm?.output_tokens_per_task ?? fallback?.outputTokensPerTask ?? null;
  const latency = arm?.median_latency_sec ?? fallback?.medianLatencySec ?? null;
  if (score === null || inputTokens === null || outputTokens === null || latency === null) return null;
  const price = listedPrice(priceModel);
  const reportedCost = arm?.estimated_cost_per_1k_tasks ?? fallback?.costPer1kTasks ?? null;
  return {
    id,
    title,
    model,
    score,
    inputTokensPerTask: inputTokens,
    outputTokensPerTask: outputTokens,
    medianLatencySec: latency,
    estimatedCostPer1kTasks:
      reportedCost ?? (price ? estimateCostPer1kTasks(inputTokens, outputTokens, price) : null),
    evalJobName: arm?.eval_job ?? '—',
  };
};

/**
 * The comparison arms, baseline first. Each figure comes from the workflow's results, or from the
 * arm's Intake evaluation when the workflow reports none. The original arm is missing when the workflow
 * started from existing trajectory filesets, so the list can hold two arms. Null until every arm has
 * its score, tokens and latency. Cost is the platform's when it reports one, else tokens at the model's list price,
 * and null for a model with no listed price.
 */
export const toArms = (
  workflow: DistillationWorkflow,
  metrics: Partial<Record<ArmId, ArmMetrics>> = {}
): MockArmResult[] | null => {
  const results = workflow.results;
  if (!results) return null;
  const studentModel = workflow.inputs.student_model ?? 'student model';
  const arms = [
    results.original
      ? toArm(
          'original',
          'Baseline',
          workflow.inputs.teacher?.model ?? 'source agent',
          results.original,
          metrics.original,
          workflow.inputs.teacher?.model
        )
      : null,
    toArm('base', 'Model swap', studentModel, results.base, metrics.base, studentModel),
    toArm('distilled', 'After distillation', `${studentModel} + adapter`, results.distilled, metrics.distilled, studentModel),
  ];
  const present = arms.filter((arm): arm is MockArmResult => arm !== null);
  const expected = results.original ? 3 : 2;
  return present.length === expected ? present : null;
};

export const agentNameOf = (workflow: DistillationWorkflow): string =>
  workflow.source_agent.split('/').pop() ?? workflow.source_agent;
