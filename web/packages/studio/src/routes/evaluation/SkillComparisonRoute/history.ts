// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { evalsListAgentEvaluateJobs } from '@nemo/sdk/generated/evals/evals-plugin-agent-eval-jobs-routes';
import {
  evalsGetTasksetRunTask,
  evalsListTasksetRunTasks,
} from '@nemo/sdk/generated/evals/evals-plugin-taskset-run-routes';
import type { AgentEvaluateJob, TasksetRunStatus } from '@nemo/sdk/generated/evals/schema';
import { customFetch } from '@nemo/sdk/generated/fetchers/evals';
import { jobsGetJob } from '@nemo/sdk/generated/platform/jobs';
import {
  parseReport,
  type ComparisonReport,
} from '@studio/routes/evaluation/SkillComparisonRoute/report';
import { z } from 'zod';

const MAX_BUNDLE_BYTES = 32 * 1024 * 1024;
const taskJobSchema = z.object({
  spec: z.object({
    spec: z.object({ target: z.object({ skills: z.array(z.string()).nullish() }) }),
    environment: z.object({
      agent: z.object({ image_ref: z.string() }),
      verifier: z.object({ image_ref: z.string() }),
    }),
  }),
});

export async function readScoringBundle(blob: Blob) {
  if (blob.size > MAX_BUNDLE_BYTES) throw new Error('Scoring bundle exceeds 32 MiB.');
  const reader = blob.stream().pipeThrough(new DecompressionStream('gzip')).getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const result = await reader.read();
      if (result.done) break;
      size += result.value.byteLength;
      if (size > MAX_BUNDLE_BYTES) throw new Error('Expanded scoring bundle exceeds 32 MiB.');
      chunks.push(result.value);
    }
  } finally {
    await reader.cancel();
  }
  const bytes = new Uint8Array(size);
  let position = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, position);
    position += chunk.length;
  }
  const decode = (data: Uint8Array) => new TextDecoder().decode(data).replace(/\0.*$/s, '');
  const files = new Map<string, string>();
  for (let offset = 0; offset + 512 <= size; ) {
    const name = decode(bytes.subarray(offset, offset + 100));
    if (!name) break;
    const length = Number.parseInt(decode(bytes.subarray(offset + 124, offset + 136)).trim(), 8);
    if (!Number.isSafeInteger(length) || length < 0 || offset + 512 + length > size)
      throw new Error('Invalid scoring archive member.');
    const base = name.split('/').pop() ?? '';
    if (['summary.json', 'scores.jsonl', 'trials.jsonl'].includes(base)) {
      if (files.has(base)) throw new Error('Ambiguous scoring archive members.');
      files.set(base, decode(bytes.subarray(offset + 512, offset + 512 + length)));
    }
    offset += 512 + Math.ceil(length / 512) * 512;
  }
  const summary = files.get('summary.json');
  const scores = files.get('scores.jsonl');
  if (!summary || !scores) throw new Error('Scoring bundle is missing summary or trial scores.');
  return {
    summary: JSON.parse(summary) as unknown,
    scores: scores
      .split('\n')
      .filter((line) => line.trim())
      .map((line): unknown => JSON.parse(line)),
  };
}

async function loadArm(workspace: string, run: TasksetRunStatus, scoringJobs: AgentEvaluateJob[]) {
  const members = await evalsListTasksetRunTasks(workspace, run.run_id, { page_size: 2 });
  if (members.data.length !== 1 || (members.pagination?.total_results ?? 1) !== 1)
    throw new Error(
      'Saved comparisons currently require one task and one trial per arm. Import a comparison report for other layouts.'
    );
  const member = members.data[0];
  const task = await evalsGetTasksetRunTask(workspace, run.run_id, member.task_index);
  if (!member.job || task.trials?.length !== 1)
    throw new Error('This run does not retain exactly one live trial.');
  const trialId = task.trials[0].trial;
  const matching = scoringJobs.filter(
    (job) =>
      job.status === 'completed' &&
      job.spec.trials?.length === 1 &&
      job.spec.trials[0].id === trialId
  );
  if (!matching.length)
    throw new Error('No completed scoring replay matches this run in the latest 100 scoring jobs.');
  const scoring = matching[0];
  const trial = scoring.spec.trials?.[0];
  const [job, bundle] = await Promise.all([
    jobsGetJob(workspace, member.job),
    customFetch<Blob>({
      url: `/apis/evals/v2/workspaces/${encodeURIComponent(workspace)}/agent-evaluate/jobs/${encodeURIComponent(scoring.name)}/results/agent-eval-results/download`,
      method: 'GET',
      responseType: 'blob',
      maxContentLength: MAX_BUNDLE_BYTES,
    }).then(readScoringBundle),
  ]);
  const stored = taskJobSchema.parse(job);
  return {
    skills: stored.spec.spec.target.skills ?? [],
    arm: {
      run_id: run.run_id,
      job_name: member.job,
      reward: task.trials[0].reward,
      agent_image_ref: stored.spec.environment.agent.image_ref,
      verifier_image_ref: stored.spec.environment.verifier.image_ref,
      bundle: 'Retained execution artifacts',
      activation: z.record(z.boolean()).safeParse(trial?.metadata?.skill_activation).success
        ? z.record(z.boolean()).parse(trial?.metadata?.skill_activation)[
            (stored.spec.spec.target.skills?.[0] ?? '').split('/').pop()?.split('#')[0] ?? ''
          ]
        : undefined,
      measurements: trial?.measurements,
      scoring_job: { name: scoring.name },
      replay: { bundle: 'Retained scoring artifacts', ...bundle },
    },
  };
}

export async function loadSavedComparison(
  workspace: string,
  baseline: TasksetRunStatus,
  treatment: TasksetRunStatus
): Promise<ComparisonReport> {
  if (baseline.run_id === treatment.run_id) throw new Error('Choose two distinct runs.');
  if (baseline.taskset_ref !== treatment.taskset_ref)
    throw new Error('Choose runs with the same pinned taskset.');
  const scoring = await evalsListAgentEvaluateJobs(workspace, {
    page_size: 100,
    sort: '-created_at',
  });
  const [before, after] = await Promise.all([
    loadArm(workspace, baseline, scoring.data),
    loadArm(workspace, treatment, scoring.data),
  ]);
  if (before.skills.length || after.skills.length !== 1)
    throw new Error('Choose a control without skills and a variant with exactly one pinned skill.');
  return parseReport(
    JSON.stringify({
      context: 'Saved Helix executions',
      workspace,
      taskset_ref: treatment.taskset_ref,
      skill_ref: after.skills[0],
      arms: { baseline: before.arm, 'with-skill': after.arm },
    })
  );
}
