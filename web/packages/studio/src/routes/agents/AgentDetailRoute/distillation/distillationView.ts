// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DistillationWorkflow,
  WorkflowArm,
  WorkflowStage,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import type { MockArmResult } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import type { MockStage } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationWorkflow';

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
  arm: WorkflowArm | null
): MockArmResult | null => {
  if (
    !arm ||
    arm.score === null ||
    arm.input_tokens_per_task === null ||
    arm.output_tokens_per_task === null ||
    arm.median_latency_sec === null ||
    arm.estimated_cost_per_1k_tasks === null
  ) {
    return null;
  }
  return {
    id,
    title,
    model,
    score: arm.score,
    inputTokensPerTask: arm.input_tokens_per_task,
    outputTokensPerTask: arm.output_tokens_per_task,
    medianLatencySec: arm.median_latency_sec,
    estimatedCostPer1kTasks: arm.estimated_cost_per_1k_tasks,
    evalJobName: arm.eval_job ?? '—',
  };
};

/**
 * The comparison arms, baseline first. The original arm is missing when the workflow started from
 * existing trajectory filesets instead of running the teacher, so the list can hold two arms.
 * Null until every arm has all of its figures.
 */
export const toArms = (workflow: DistillationWorkflow): MockArmResult[] | null => {
  const results = workflow.results;
  if (!results) return null;
  const studentModel = workflow.inputs.student_model ?? 'student model';
  const arms = [
    results.original
      ? toArm('original', 'Baseline', workflow.inputs.teacher?.model ?? 'source agent', results.original)
      : null,
    toArm('base', 'Model swap', studentModel, results.base),
    toArm('distilled', 'After distillation', `${studentModel} + adapter`, results.distilled),
  ];
  const present = arms.filter((arm): arm is MockArmResult => arm !== null);
  const expected = results.original ? 3 : 2;
  return present.length === expected ? present : null;
};

export const agentNameOf = (workflow: DistillationWorkflow): string =>
  workflow.source_agent.split('/').pop() ?? workflow.source_agent;
