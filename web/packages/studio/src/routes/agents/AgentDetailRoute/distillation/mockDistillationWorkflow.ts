// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildMockResults,
  HELD_OUT_TASKS,
  type MockArmResult,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import { useEffect, useState } from 'react';

export type MockStageStatus = 'pending' | 'active' | 'completed' | 'failed';

export interface MockStageMetric {
  label: string;
  value: string;
}

export interface MockStage {
  id: 'prepare' | 'customization' | 'eval-original' | 'eval-base' | 'eval-distilled';
  title: string;
  description: string;
  status: MockStageStatus;
  progress: number;
  /** The platform job the stage runs as, once the backend workflow reports it. */
  jobRef?: { kind: string; name: string };
  metrics: MockStageMetric[];
}

export interface MockWorkflow {
  name: string;
  agentName: string;
  studentModel: string;
  runCount: number;
  exampleCount: number;
  startedAt: number;
  status: MockStageStatus;
  stages: MockStage[];
}

export interface MockDistillationRequest {
  name: string;
  agentName: string;
  studentModel: string;
  runCount: number;
  exampleCount: number;
}

const PREPARE_SECONDS = 2;
const CUSTOMIZATION_SECONDS = 10;
const EPOCHS = 3;
const EVAL_ORIGINAL = { start: PREPARE_SECONDS, duration: 6 };
const EVAL_BASE = { start: PREPARE_SECONDS, duration: 8 };
const EVAL_DISTILLED = { start: PREPARE_SECONDS + CUSTOMIZATION_SECONDS, duration: 7 };
const TOTAL_SECONDS = EVAL_DISTILLED.start + EVAL_DISTILLED.duration;

const requests = new Map<string, MockDistillationRequest & { startedAt: number }>();

export const startMockDistillation = (request: MockDistillationRequest) => {
  requests.set(request.name, { ...request, startedAt: Date.now() });
};

const stageStatus = (elapsed: number, start: number, duration: number): MockStageStatus =>
  elapsed >= start + duration ? 'completed' : elapsed >= start ? 'active' : 'pending';

const stageProgress = (elapsed: number, start: number, duration: number): number =>
  Math.round(Math.min(1, Math.max(0, (elapsed - start) / duration)) * 100);

export const buildMockWorkflow = (
  request: MockDistillationRequest & { startedAt: number },
  now: number
): MockWorkflow => {
  const elapsed = (now - request.startedAt) / 1000;
  const customizationStart = PREPARE_SECONDS;

  const prepareProgress = stageProgress(elapsed, 0, PREPARE_SECONDS);
  const customizationProgress = stageProgress(elapsed, customizationStart, CUSTOMIZATION_SECONDS);
  const epoch = Math.min(EPOCHS, Math.floor((customizationProgress / 100) * EPOCHS) + 1);
  const results = buildMockResults(request.name, request.studentModel);

  const evalStage = (
    id: MockStage['id'],
    title: string,
    description: string,
    timing: { start: number; duration: number },
    result: MockArmResult
  ): MockStage => {
    const progress = stageProgress(elapsed, timing.start, timing.duration);
    return {
      id,
      title,
      description,
      status: stageStatus(elapsed, timing.start, timing.duration),
      progress,
      jobRef: { kind: 'Evaluation job', name: result.evalJobName },
      metrics: [
        { label: 'Held-out tasks', value: `${Math.round((progress / 100) * HELD_OUT_TASKS)} / ${HELD_OUT_TASKS}` },
        { label: 'Score', value: progress >= 100 ? result.score.toFixed(2) : '—' },
      ],
    };
  };

  const stages: MockStage[] = [
    {
      id: 'prepare',
      title: 'Prepare training data',
      description: `Split the tasks, keeping ${HELD_OUT_TASKS} held out, and convert passing trials from the selected runs into training examples.`,
      status: stageStatus(elapsed, 0, PREPARE_SECONDS),
      progress: prepareProgress,
      metrics: [
        { label: 'Runs', value: String(request.runCount) },
        {
          label: 'Examples',
          value: String(Math.round((prepareProgress / 100) * request.exampleCount)),
        },
        { label: 'Held out', value: `${HELD_OUT_TASKS} tasks` },
      ],
    },
    {
      id: 'customization',
      title: 'Fine-tune student model',
      description: `Train ${request.studentModel} on the prepared examples.`,
      status: stageStatus(elapsed, customizationStart, CUSTOMIZATION_SECONDS),
      progress: customizationProgress,
      jobRef: { kind: 'Customization job', name: `${request.name}-sft` },
      metrics: [
        { label: 'Epoch', value: `${customizationProgress > 0 ? epoch : 0} / ${EPOCHS}` },
        {
          label: 'Train loss',
          value: customizationProgress > 0 ? (1.9 - customizationProgress * 0.012).toFixed(2) : '—',
        },
      ],
    },
    evalStage(
      'eval-original',
      'Evaluate original agent',
      'Baseline: the agent as it runs today, on the held-out tasks.',
      EVAL_ORIGINAL,
      results[0]
    ),
    evalStage(
      'eval-base',
      'Evaluate model swap (no training)',
      'The same agent with the student model and no training, on the held-out tasks.',
      EVAL_BASE,
      results[1]
    ),
    evalStage(
      'eval-distilled',
      'Evaluate distilled model',
      'The same agent with the fine-tuned student model, on the held-out tasks.',
      EVAL_DISTILLED,
      results[2]
    ),
  ];

  return {
    name: request.name,
    agentName: request.agentName,
    studentModel: request.studentModel,
    runCount: request.runCount,
    exampleCount: request.exampleCount,
    startedAt: request.startedAt,
    status: elapsed >= TOTAL_SECONDS ? 'completed' : 'active',
    stages,
  };
};

/**
 * Stand-in for polling the backend distillation workflow. Replays a fixed timeline from the moment
 * the distillation was started, and falls back to a fresh run when the page is opened directly.
 */
export const useMockDistillationWorkflow = (name: string, agentName?: string): MockWorkflow => {
  // TODO(api): poll the workflow entity until it reaches a terminal status (proposed shape).
  // const { data: workflow } = useAgentDistillationsGet(workspace, name, {
  //   query: {
  //     refetchInterval: (query) =>
  //       TERMINAL_STATUSES.has(query.state.data?.status) ? false : JOB_POLLING_INTERVAL_MS,
  //   },
  // });
  // Each stage maps to a job reference: workflow.stages[].job_ref -> { kind, name }, with progress
  // and metrics read from that job's own status endpoint.
  const [now, setNow] = useState(() => Date.now());

  if (!requests.has(name)) {
    requests.set(name, {
      name,
      agentName: agentName ?? 'agent',
      studentModel: 'default/nemotron-35-lightning-30b-a3b-bf16',
      runCount: 3,
      exampleCount: 272,
      startedAt: Date.now(),
    });
  }
  const request = requests.get(name);

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  if (!request) throw new Error(`Missing mock distillation request for ${name}`);
  return buildMockWorkflow(request, now);
};
