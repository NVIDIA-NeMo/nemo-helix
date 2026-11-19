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
  fetchAgentEvalBundle,
  fetchAgentEvalResult,
  type AgentEvalScoreRow,
} from '@studio/api/evaluation/agent-evaluations';
import {
  parseReport,
  type ComparisonReport,
} from '@studio/routes/evaluation/SkillComparisonRoute/report';
import { z } from 'zod';

const MAX_BUNDLE_BYTES = 32 * 1024 * 1024;
const packageDiagnostic = z.object({ details: z.object({ metric: z.string() }) });
const taskJobSchema = z.object({
  spec: z.object({
    spec: z.object({ target: z.object({ skills: z.array(z.string()).nullish() }) }),
    environment: z.object({
      agent: z.object({ image_ref: z.string() }),
      verifier: z.object({ image_ref: z.string() }).nullish(),
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

async function loadExecution(workspace: string, run: TasksetRunStatus) {
  const members = await evalsListTasksetRunTasks(workspace, run.run_id, { page_size: 2 });
  if (members.data.length !== 1 || (members.pagination?.total_results ?? 1) !== 1)
    throw new Error(
      'Saved comparisons currently require one task and one trial per arm. Import a comparison report for other layouts.'
    );
  const member = members.data[0];
  const task = await evalsGetTasksetRunTask(workspace, run.run_id, member.task_index);
  if (!member.job || task.trials?.length !== 1)
    throw new Error('This run does not retain exactly one live trial.');
  const stored = taskJobSchema.parse(await jobsGetJob(workspace, member.job));
  return {
    run,
    trial: task.trials[0],
    jobName: member.job,
    skills: stored.spec.spec.target.skills ?? [],
    agentImage: stored.spec.environment.agent.image_ref,
    verifierImage: stored.spec.environment.verifier?.image_ref ?? null,
  };
}

type SavedExecution = Awaited<ReturnType<typeof loadExecution>>;

function findReplay(execution: SavedExecution, jobs: AgentEvaluateJob[]) {
  return jobs.find(
    (job) =>
      job.status === 'completed' &&
      job.spec.trials?.length === 1 &&
      job.spec.trials[0].id === execution.trial.trial
  );
}

async function loadReplay(workspace: string, scoring: AgentEvaluateJob) {
  const bundle = await customFetch<Blob>({
    url: `/apis/evals/v2/workspaces/${encodeURIComponent(workspace)}/agent-evaluate/jobs/${encodeURIComponent(scoring.name)}/results/agent-eval-results/download`,
    method: 'GET',
    responseType: 'blob',
    maxContentLength: MAX_BUNDLE_BYTES,
  }).then(readScoringBundle);
  return {
    trial: scoring.spec.trials?.[0],
    scoringJob: { name: scoring.name },
    replay: { bundle: 'Retained scoring artifacts', ...bundle },
  };
}

async function loadNative(workspace: string, execution: SavedExecution, signal: AbortSignal) {
  const runId = execution.run.run_id;
  const result = await fetchAgentEvalResult(workspace, runId, signal);
  if (!result?.summary) throw new Error(`Run ${runId} has no saved scoring results.`);
  const bundle = await fetchAgentEvalBundle(workspace, result.bundle_ref, signal);
  if (!bundle) throw new Error(`Run ${runId} has no readable scoring bundle.`);
  const trial = bundle.trials.find(({ id }) => id === execution.trial.trial);
  if (!trial) throw new Error(`Run ${runId} has no matching trial in its scoring bundle.`);
  const scores = bundle.scores.filter(({ trial_id }) => trial_id === trial.id);
  const { packageMetrics, trialScores } = projectNativeScores(scores);
  return {
    trial,
    packageMetrics,
    replay: {
      bundle: result.bundle_ref,
      source: 'run' as const,
      summary: {
        ...result.summary,
        scores: {
          scores: result.scores.scores.filter(
            ({ name }) => !name.startsWith('harbor_skill_metrics.package.')
          ),
        },
        metric_coverage: {
          ...result.summary.metric_coverage,
          ...(result.summary.metric_coverage?.harbor_skill_metrics && {
            harbor_skill_metrics: Object.fromEntries(
              Object.entries(result.summary.metric_coverage.harbor_skill_metrics).filter(
                ([name]) => !name.startsWith('package.')
              )
            ),
          }),
        },
      },
      scores: trialScores,
    },
  };
}

function projectNativeScores(scores: AgentEvalScoreRow[]) {
  interface PackageResult {
    outputs: NonNullable<AgentEvalScoreRow['outputs']>;
    diagnostics: unknown[];
  }
  const packageMetrics = new Map<string, PackageResult>();
  const packageMetric = (key: string) => {
    let result = packageMetrics.get(key);
    if (!result) {
      result = { outputs: [], diagnostics: [] };
      packageMetrics.set(key, result);
    }
    return result;
  };
  const trialScores = scores.map((score) => {
    const outputs = score.outputs ?? [];
    if (score.metric_type !== 'harbor_skill_metrics') return { ...score, outputs };
    const diagnostics = score.diagnostics ?? [];
    for (const output of outputs) {
      const separator = output.name.lastIndexOf('.');
      if (!output.name.startsWith('package.') || separator <= 'package.'.length) continue;
      const key = output.name.slice(0, separator);
      packageMetric(key).outputs.push({ ...output, name: output.name.slice(separator + 1) });
    }
    const trialDiagnostics = diagnostics.filter((diagnostic) => {
      const metric = packageDiagnostic.safeParse(diagnostic);
      if (!metric.success || !metric.data.details.metric.startsWith('package.')) return true;
      packageMetric(metric.data.details.metric).diagnostics.push(diagnostic);
      return false;
    });
    return {
      ...score,
      outputs: outputs.filter(({ name }) => !name.startsWith('package.')),
      diagnostics: trialDiagnostics,
    };
  });
  return { packageMetrics: Object.fromEntries(packageMetrics), trialScores };
}

function toArm(
  execution: SavedExecution,
  results: Awaited<ReturnType<typeof loadReplay>> | Awaited<ReturnType<typeof loadNative>>
) {
  const activation = z.record(z.boolean()).safeParse(results.trial?.metadata?.skill_activation);
  const skillName =
    execution.skills.length === 1 ? execution.skills[0].split('#')[0].split('/').pop() : undefined;
  return {
    run_id: execution.run.run_id,
    job_name: execution.jobName,
    reward: execution.trial.reward ?? null,
    evidence_error: execution.trial.error,
    agent_image_ref: execution.agentImage,
    verifier_image_ref: execution.verifierImage,
    skill_refs: execution.skills,
    bundle: 'Retained execution artifacts',
    activation: activation.success && skillName ? activation.data[skillName] : undefined,
    measurements: results.trial?.measurements ?? undefined,
    scoring_job: 'scoringJob' in results ? results.scoringJob : undefined,
    replay: results.replay,
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
  const [before, after, scoring] = await Promise.all([
    loadExecution(workspace, baseline),
    loadExecution(workspace, treatment),
    evalsListAgentEvaluateJobs(workspace, { page_size: 100, sort: '-created_at' }),
  ]);
  if (before.skills.length || !after.skills.length)
    throw new Error(
      'Choose a control without skills and a variant with at least one pinned skill.'
    );

  const beforeReplay = findReplay(before, scoring.data);
  const afterReplay = findReplay(after, scoring.data);
  const signal = AbortSignal.timeout(60_000);
  const [controlResults, variantResults] = await (beforeReplay && afterReplay
    ? Promise.all([loadReplay(workspace, beforeReplay), loadReplay(workspace, afterReplay)])
    : Promise.all([loadNative(workspace, before, signal), loadNative(workspace, after, signal)]));
  return parseReport(
    JSON.stringify({
      context: 'Saved Helix executions',
      workspace,
      taskset_ref: treatment.taskset_ref,
      package_metrics:
        'packageMetrics' in variantResults ? variantResults.packageMetrics : undefined,
      arms: { baseline: toArm(before, controlResults), 'with-skill': toArm(after, variantResults) },
    })
  );
}
