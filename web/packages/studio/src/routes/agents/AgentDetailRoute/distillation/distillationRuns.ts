// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsListDeployments } from '@nemo/sdk/generated/agents/agent-deployments';
import {
  evalsGetAgentEvaluateJobResult,
  evalsListAgentEvaluateJobs,
} from '@nemo/sdk/generated/evals/evals-plugin-agent-eval-jobs-routes';
import type { AgentEvaluateJob } from '@nemo/sdk/generated/evals/schema';
import { filesCreateFileset, filesUploadFile } from '@nemo/sdk/generated/platform/files';
import {
  DISTILLATION_MOCK,
  MOCK_EVAL_RUNS,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationMock';
import { useQuery } from '@tanstack/react-query';

const JOB_FILESET_PREFIX = 'job-fileset-';
const FILESET_NAME_MAX = 63;
const BUNDLE_RESULT = 'agent-eval-results';
const EVAL_CONFIG_FILE = 'tasks.json';

export interface EvalRun {
  name: string;
  createdAt: string | undefined;
  taskCount: number;
  /** The tasks as the job ran them; only their ids are read here. */
  tasks: { id: string }[];
}

const bareName = (ref: string): string => ref.split('/').pop() ?? ref;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

/** The registered agent an eval job ran, when its target is one. */
const targetAgent = (job: AgentEvaluateJob): string | undefined => {
  const target: unknown = 'target' in job.spec ? job.spec.target : undefined;
  if (!isRecord(target) || !isRecord(target.source)) return undefined;
  return typeof target.source.agent === 'string' ? target.source.agent : undefined;
};

const toRun = (job: AgentEvaluateJob): EvalRun | undefined => {
  if (job.status !== 'completed' || !('tasks' in job.spec) || job.spec.tasks.length === 0) {
    return undefined;
  }
  return {
    name: job.name,
    createdAt: job.created_at,
    taskCount: job.spec.tasks.length,
    tasks: job.spec.tasks,
  };
};

/** The agent's completed eval jobs, newest first: the runs whose trajectories can train a student. */
export const useAgentEvalRuns = (workspace: string, agentName: string) =>
  useQuery({
    queryKey: ['distillation-eval-runs', workspace, agentName],
    queryFn: async (): Promise<EvalRun[]> => {
      if (DISTILLATION_MOCK) return MOCK_EVAL_RUNS;
      const page = await evalsListAgentEvaluateJobs(workspace, { page_size: 100 });
      return page.data
        .filter((job) => {
          const agent = targetAgent(job);
          return agent !== undefined && bareName(agent) === agentName;
        })
        .flatMap((job) => toRun(job) ?? [])
        .sort((a, b) => (b.createdAt ?? '').localeCompare(a.createdAt ?? ''));
    },
  });

/**
 * The fileset a job writes its results to. Short job names give `job-fileset-<job>` as is; a longer
 * name is hashed by the platform, so it is read back from the job's result location instead.
 */
export const trajectoryFilesetOf = async (workspace: string, job: string): Promise<string> => {
  if (DISTILLATION_MOCK) return `${workspace}/job-fileset-${job}`;
  if (JOB_FILESET_PREFIX.length + job.length <= FILESET_NAME_MAX) {
    return `${workspace}/${JOB_FILESET_PREFIX}${job}`;
  }
  const result = await evalsGetAgentEvaluateJobResult(workspace, job, BUNDLE_RESULT);
  const match = /job-fileset-[A-Za-z0-9-]+/.exec(result.artifact_url);
  if (!match) throw new Error(`Could not find the results fileset of eval run ${job}.`);
  return `${workspace}/${match[0]}`;
};

/** Writes the tasks the trajectories came from as the eval config the workflow scores every model on. */
export const writeEvalConfig = async (
  workspace: string,
  agentName: string,
  runs: EvalRun[]
): Promise<string> => {
  if (DISTILLATION_MOCK) return `${workspace}/distill-evals-mock#${EVAL_CONFIG_FILE}`;
  const tasks = new Map<string, { id: string }>();
  runs.forEach((run) => run.tasks.forEach((task) => tasks.set(task.id, task)));
  const fileset = `distill-evals-${agentName}-${Date.now().toString(36)}`
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, '-')
    .slice(0, FILESET_NAME_MAX);
  await filesCreateFileset(workspace, { name: fileset, description: 'Distillation eval tasks' });
  await filesUploadFile(
    workspace,
    fileset,
    EVAL_CONFIG_FILE,
    new Blob([JSON.stringify({ tasks: [...tasks.values()] })], { type: 'application/json' })
  );
  return `${workspace}/${fileset}#${EVAL_CONFIG_FILE}`;
};

/** An image the agent already runs as, so the distilled agent's deployments use the same one. */
export const useAgentDeploymentImage = (workspace: string, agentName: string) =>
  useQuery({
    queryKey: ['distillation-agent-image', workspace, agentName],
    queryFn: async () => {
      if (DISTILLATION_MOCK) return null;
      const page = await agentsListDeployments(workspace, { page_size: 100 });
      const deployment = page.data.find(
        (candidate) =>
          candidate.agent !== undefined &&
          bareName(candidate.agent) === agentName &&
          !!candidate.image &&
          candidate.deployment_mode !== 'subprocess'
      );
      return deployment?.image && deployment.deployment_mode
        ? { image: deployment.image, mode: deployment.deployment_mode }
        : null;
    },
  });
