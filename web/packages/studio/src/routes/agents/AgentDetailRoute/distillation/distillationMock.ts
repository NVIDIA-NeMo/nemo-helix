// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  CreateWorkflowBody,
  DistillationWorkflow,
  DistillInputs,
  GraphNode,
  NodeStatus,
  RenderResponse,
  RuntimeNode,
  StageStatus,
  WorkflowStage,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import type { ArmMetrics } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationArmMetrics';
import type { ArmId } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import type { TrainingProgress } from '@studio/routes/agents/AgentDetailRoute/distillation/useTrainingProgress';

/** Serves the distillation API from the browser until the backend is ready. Set `VITE_DISTILLATION_MOCK=false` to call the real API. */
export const DISTILLATION_MOCK = import.meta.env.VITE_DISTILLATION_MOCK !== 'false';

const STORAGE_KEY = 'distillation-mock-runs';
const TRAIN_STEPS = 240;

interface StoredRun {
  workspace: string;
  name: string;
  inputs: DistillInputs;
  startedAt: number;
  cancelledAt?: number;
  failTrain?: boolean;
}

const readRuns = (): Record<string, StoredRun> => {
  try {
    return JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) ?? '{}');
  } catch {
    return {};
  }
};

const writeRuns = (runs: Record<string, StoredRun>) => {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
  } catch {
    // Without storage a reload forgets the run.
  }
};

const runKey = (workspace: string, name: string) => `${workspace}/${name}`;

interface Step {
  key: string;
  kind: string;
  seconds: number;
  after: string[];
}

const candidateList = (inputs: DistillInputs): string[] =>
  inputs.candidates && inputs.candidates.length > 0
    ? inputs.candidates.map((candidate) => candidate.name)
    : ['lora'];

const steps = (inputs: DistillInputs): Step[] => {
  const candidates = candidateList(inputs);
  const evalName = inputs.evals[0]?.name ?? 'holdout';
  const original: Step[] = inputs.teacher
    ? [{ key: `eval_original[${evalName}]`, kind: 'agent_eval', seconds: 7, after: [] }]
    : [];
  const originalKeys = original.map((step) => step.key);
  return [
    ...original,
    { key: 'prepare', kind: 'convert_atif', seconds: 5, after: originalKeys },
    { key: `eval_base[${evalName}]`, kind: 'agent_eval', seconds: 7, after: ['prepare'] },
    ...candidates.flatMap((candidate) => [
      { key: `train[${candidate}]`, kind: 'customization_job', seconds: 12, after: ['prepare'] },
      {
        key: `eval_distilled_${evalName}[${candidate}]`,
        kind: 'agent_eval',
        seconds: 7,
        after: [`train[${candidate}]`],
      },
    ]),
  ];
};

const candidateOf = (key: string): string => /\[(.+)\]$/.exec(key)?.[1] ?? 'lora';

const slug = (value: string) =>
  value
    .replace(/[^a-z0-9]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();

const windows = (all: Step[]): Record<string, { start: number; end: number }> => {
  const found: Record<string, { start: number; end: number }> = {};
  all.forEach((step) => {
    const start = Math.max(0, ...step.after.map((key) => found[key]?.end ?? 0));
    found[step.key] = { start: start + 1, end: start + 1 + step.seconds };
  });
  return found;
};

const nodeStatus = (run: StoredRun, window: { start: number; end: number }, now: number): NodeStatus => {
  const elapsed = (now - run.startedAt) / 1000;
  if (run.cancelledAt && now >= run.cancelledAt) {
    return elapsed >= window.end && !run.failTrain ? 'succeeded' : 'cancelled';
  }
  if (elapsed < window.start) return 'pending';
  return elapsed >= window.end ? 'succeeded' : 'running';
};

const STAGES: { id: string; name: string; prefix: string }[] = [
  { id: 'prepare', name: 'Prepare', prefix: 'prepare' },
  { id: 'customization', name: 'Customization', prefix: 'train' },
  { id: 'eval-original', name: 'Original eval', prefix: 'eval_original' },
  { id: 'eval-base', name: 'Base eval', prefix: 'eval_base' },
  { id: 'eval-distilled', name: 'Distilled eval', prefix: 'eval_distilled' },
];

const stageOf = (statuses: NodeStatus[]): { status: StageStatus; progress: number } => {
  if (statuses.some((status) => status === 'failed' || status === 'skipped')) {
    return { status: 'failed', progress: 100 };
  }
  if (statuses.some((status) => status === 'running')) return { status: 'active', progress: 50 };
  if (statuses.every((status) => status === 'succeeded')) return { status: 'completed', progress: 100 };
  if (statuses.some((status) => status === 'cancelled')) return { status: 'failed', progress: 100 };
  return { status: 'pending', progress: 0 };
};

const workflowOf = (run: StoredRun, now: number): DistillationWorkflow => {
  const all = steps(run.inputs);
  const spans = windows(all);
  const graph: Record<string, GraphNode> = {};
  const nodes: Record<string, RuntimeNode> = {};

  all.forEach((step) => {
    graph[step.key] = { kind: step.kind, inputs: {}, depends_on: step.after, always_run: false };
    let status = nodeStatus(run, spans[step.key], now);
    if (run.failTrain && step.kind === 'customization_job' && status === 'running') status = 'failed';
    const job = `${run.name}-${slug(step.key)}`;
    const outputs: Record<string, unknown> =
      status === 'succeeded' && step.key === 'prepare'
        ? { rows: 128, dropped_rows: 12 }
        : status === 'succeeded' && step.kind === 'customization_job'
          ? { job, model: `${run.workspace}/${run.name}-${candidateOf(step.key)}` }
          : {};
    nodes[step.key] = {
      status,
      attempt: 1,
      external: { job },
      outputs,
      error: status === 'failed' ? 'training job exited with code 1' : null,
    };
  });

  graph.experiment = { kind: 'experiment', inputs: {}, depends_on: [], always_run: false };
  nodes.experiment = {
    status: 'succeeded',
    attempt: 1,
    external: {},
    outputs: {
      experiment: run.name,
      evaluations: {
        original: `${run.name}-original`,
        base: `${run.name}-base`,
        ...Object.fromEntries(candidateList(run.inputs).map((name) => [name, `${run.name}-${name}`])),
      },
    },
    error: null,
  };

  const stages: WorkflowStage[] = STAGES.flatMap((stage) => {
    const keys = all.filter((step) => step.key.startsWith(stage.prefix)).map((step) => step.key);
    if (keys.length === 0) return [];
    const { status, progress } = stageOf(keys.map((key) => nodes[key].status));
    const first = nodes[keys[0]];
    return [
      {
        id: stage.id,
        name: stage.name,
        status,
        progress,
        metrics: Object.keys(first.outputs).length > 0 ? first.outputs : {},
        job_ref: first.external.job ?? null,
      },
    ];
  });

  const evalNodes = all.filter((step) => step.kind === 'agent_eval').map((step) => nodes[step.key]);
  const finished = all.every((step) => nodes[step.key].status === 'succeeded');
  const anyFailed = all.some((step) => nodes[step.key].status === 'failed');
  const cancelled = run.cancelledAt !== undefined && now >= run.cancelledAt && !finished;
  const arm = (job: string | undefined) => ({
    score: null,
    input_tokens_per_task: null,
    output_tokens_per_task: null,
    median_latency_sec: null,
    estimated_cost_per_1k_tasks: null,
    eval_job: job ?? null,
    filesets: [],
  });
  const evalJob = (prefix: string) => all.find((step) => step.key.startsWith(prefix))?.key;

  return {
    name: run.name,
    workspace: run.workspace,
    created_at: new Date(run.startedAt).toISOString(),
    template: 'distillation',
    status: finished ? 'completed' : anyFailed ? 'failed' : cancelled ? 'cancelled' : 'running',
    source_agent: run.inputs.source_agent,
    inputs: run.inputs,
    fileset: `${run.name}-files`,
    graph,
    nodes,
    experiment: run.name,
    stages,
    results:
      finished && evalNodes.length > 0
        ? {
            original: run.inputs.teacher ? arm(nodes[evalJob('eval_original') ?? '']?.external.job) : null,
            base: arm(nodes[evalJob('eval_base') ?? '']?.external.job),
            distilled: arm(nodes[evalJob('eval_distilled') ?? '']?.external.job),
            prices: {},
          }
        : null,
  };
};

const delay = <T>(value: T): Promise<T> => new Promise((resolve) => setTimeout(() => resolve(value), 250));

export const mockRender = (inputs: DistillInputs): Promise<RenderResponse> => {
  const all = steps(inputs);
  return delay({
    graph: Object.fromEntries(
      all.map((step) => [
        step.key,
        { kind: step.kind, inputs: {}, depends_on: step.after, always_run: false },
      ])
    ),
    counts: {
      nodes: all.length + 6,
      candidates: inputs.candidates?.length ?? 1,
      passing_trials: 128,
      estimated_tokens: 480_000,
      gpu_hours: 3,
      parallel_jobs: Math.max(1, Math.floor((inputs.gpu_budget ?? 1) / (inputs.deployment?.gpu ?? 1))),
    },
  });
};

export const mockCreate = (workspace: string, body: CreateWorkflowBody): Promise<DistillationWorkflow> => {
  const name = body.name ?? `distill-${Math.random().toString(16).slice(2, 10)}`;
  const candidates =
    body.inputs.candidates && body.inputs.candidates.length > 1
      ? body.inputs.candidates
      : [
          ...(body.inputs.candidates ?? [{ name: 'lora', finetuning_type: 'lora' as const }]),
          { name: 'full', finetuning_type: 'all_weights' as const, gpu: 1 },
          { name: 'lora-r64', finetuning_type: 'lora' as const, gpu: 1 },
          { name: 'lora-r128', finetuning_type: 'lora' as const, gpu: 1 },
        ];
  const run: StoredRun = {
    workspace,
    name,
    inputs: { ...body.inputs, candidates },
    startedAt: Date.now(),
  };
  writeRuns({ ...readRuns(), [runKey(workspace, name)]: run });
  return delay(workflowOf(run, Date.now()));
};

export const mockGet = (workspace: string, name: string): Promise<DistillationWorkflow> => {
  const run = readRuns()[runKey(workspace, name)];
  return run ? Promise.resolve(workflowOf(run, Date.now())) : Promise.reject(new Error('Workflow not found'));
};

export const mockCancel = async (workspace: string, name: string): Promise<DistillationWorkflow> => {
  const runs = readRuns();
  const run = runs[runKey(workspace, name)];
  if (!run) throw new Error('Workflow not found');
  run.cancelledAt = Date.now();
  writeRuns(runs);
  return workflowOf(run, Date.now());
};

export const mockRetry = async (workspace: string, name: string): Promise<DistillationWorkflow> => {
  const runs = readRuns();
  const run = runs[runKey(workspace, name)];
  if (!run) throw new Error('Workflow not found');
  run.failTrain = false;
  run.startedAt = Date.now();
  writeRuns(runs);
  return workflowOf(run, Date.now());
};

const MOCK_ARM_METRICS: Record<string, ArmMetrics> = {
  original: {
    score: 0.88,
    inputTokensPerTask: 3200,
    outputTokensPerTask: 450,
    medianLatencySec: 9.8,
    costPer1kTasks: null,
    model: 'default/glm-5-3-flash',
  },
  base: {
    score: 0.54,
    inputTokensPerTask: 3400,
    outputTokensPerTask: 700,
    medianLatencySec: 3.1,
    costPer1kTasks: null,
  },
  lora: {
    score: 0.85,
    inputTokensPerTask: 3150,
    outputTokensPerTask: 430,
    medianLatencySec: 3.4,
    costPer1kTasks: null,
  },
  'lora-r64': {
    score: 0.82,
    inputTokensPerTask: 3180,
    outputTokensPerTask: 450,
    medianLatencySec: 3.3,
    costPer1kTasks: null,
  },
  'lora-r128': {
    score: 0.86,
    inputTokensPerTask: 3120,
    outputTokensPerTask: 420,
    medianLatencySec: 3.5,
    costPer1kTasks: null,
  },
  full: {
    score: 0.87,
    inputTokensPerTask: 3100,
    outputTokensPerTask: 410,
    medianLatencySec: 3.6,
    costPer1kTasks: null,
  },
};

export const mockArmMetrics = (arm: ArmId): Promise<ArmMetrics> =>
  delay(MOCK_ARM_METRICS[arm] ?? MOCK_ARM_METRICS.lora);

/** Training progress for the run's train job, from how far into its window the clock is. */
export const mockTrainingProgress = (job: string | undefined): TrainingProgress | undefined => {
  const run = Object.values(readRuns()).find((candidate) => job?.startsWith(`${candidate.name}-train`));
  if (!run) return undefined;
  const all = steps(run.inputs);
  const trainKey = all.find((step) => step.kind === 'customization_job' && job === `${run.name}-${slug(step.key)}`)?.key;
  const span = trainKey ? windows(all)[trainKey] : undefined;
  if (!span) return undefined;
  const fraction = Math.min(1, Math.max(0, ((Date.now() - run.startedAt) / 1000 - span.start) / (span.end - span.start)));
  const step = Math.round(fraction * TRAIN_STEPS);
  const loss = 2.1 - 1.4 * fraction;
  return {
    percent: fraction * 100,
    detail: `training · step ${step}/${TRAIN_STEPS} · epoch ${Math.min(3, Math.floor(fraction * 3) + 1)} · loss ${loss.toFixed(3)}`,
  };
};

export const MOCK_EVAL_RUNS = [
  { name: 'triage-eval-smoke', createdAt: '2026-11-17T10:00:00Z', taskCount: 12 },
  { name: 'triage-eval-full', createdAt: '2026-11-18T09:30:00Z', taskCount: 40 },
].map((run) => ({
  ...run,
  tasks: Array.from({ length: run.taskCount }, (_, index) => ({ id: `${run.name}-task-${index}` })),
}));

export const MOCK_MODEL_NAMES = ['llama-3-3-70b-instruct', 'nemotron-3-5-lightning-30b-a3b', 'qwen3-8b'];
