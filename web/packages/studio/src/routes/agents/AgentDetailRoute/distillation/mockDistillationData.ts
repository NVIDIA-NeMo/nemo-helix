// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface MockTraceEvalRun {
  name: string;
  createdAt: string;
  taskCount: number;
  passedCount: number;
  hasTrajectories: boolean;
}

export interface MockStudentModel {
  value: string;
  label: string;
  params: string;
}

// TODO(api): list the agent's task evaluations that captured trajectories (proposed shape).
// const { data } = useAgentEvaluateListJobs(workspace, { agent: agentName, has_trajectories: true });
// Map each job to { name, createdAt, taskCount, passedCount } from its agent-eval-results record.
export const MOCK_TRACE_EVAL_RUNS: MockTraceEvalRun[] = [
  {
    name: 'email-triage-fabric-eval-3',
    createdAt: '2026-11-17T16:20:00Z',
    taskCount: 120,
    passedCount: 104,
    hasTrajectories: true,
  },
  {
    name: 'email-triage-fabric-eval-2',
    createdAt: '2026-11-14T09:05:00Z',
    taskCount: 120,
    passedCount: 97,
    hasTrajectories: true,
  },
  {
    name: 'email-triage-fabric-eval-1',
    createdAt: '2026-11-10T21:42:00Z',
    taskCount: 60,
    passedCount: 41,
    hasTrajectories: true,
  },
  {
    name: 'email-triage-http-eval-7',
    createdAt: '2026-11-09T13:30:00Z',
    taskCount: 80,
    passedCount: 71,
    hasTrajectories: false,
  },
];

// TODO(api): swap for the models search the judge picker uses (useModelSearch, chat-capable only).
export const MOCK_STUDENT_MODELS: MockStudentModel[] = [
  {
    value: 'default/nemotron-35-lightning-30b-a3b-bf16',
    label: 'Nemotron 3.5 Lightning 30B A3B',
    params: '30B',
  },
  { value: 'default/qwen3-1-7b', label: 'Qwen3-1.7B', params: '1.7B' },
  { value: 'default/qwen3-4b', label: 'Qwen3-4B', params: '4B' },
  { value: 'default/llama-3-2-3b-instruct', label: 'Llama 3.2 3B Instruct', params: '3B' },
  { value: 'default/nemotron-nano-9b-v2', label: 'Nemotron Nano 9B v2', params: '9B' },
];

export const DEFAULT_STUDENT_MODEL = MOCK_STUDENT_MODELS.find((model) =>
  /nemotron.*30b/i.test(model.value)
)?.value;
