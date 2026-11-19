// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { evalsListAgentEvaluateJobs } from '@nemo/sdk/generated/evals/evals-plugin-agent-eval-jobs-routes';
import {
  evalsGetTasksetRunTask,
  evalsListTasksetRunTasks,
} from '@nemo/sdk/generated/evals/evals-plugin-taskset-run-routes';
import type { TasksetRunStatus } from '@nemo/sdk/generated/evals/schema';
import { customFetch } from '@nemo/sdk/generated/fetchers/evals';
import { jobsGetJob } from '@nemo/sdk/generated/platform/jobs';
import {
  fetchAgentEvalBundle,
  fetchAgentEvalResult,
} from '@studio/api/evaluation/agent-evaluations';
import {
  readScoringBundle,
  loadSavedComparison,
} from '@studio/routes/evaluation/SkillComparisonRoute/history';
import { Blob as NodeBlob } from 'node:buffer';
import { gzipSync } from 'node:zlib';

function archive(files: Record<string, string>) {
  const parts: Buffer[] = [];
  for (const [name, body] of Object.entries(files)) {
    const header = Buffer.alloc(512);
    header.write(name);
    header.write(Buffer.byteLength(body).toString(8).padStart(11, '0'), 124);
    parts.push(
      header,
      Buffer.from(body),
      Buffer.alloc((512 - (Buffer.byteLength(body) % 512)) % 512)
    );
  }
  parts.push(Buffer.alloc(1024));
  return new NodeBlob([gzipSync(Buffer.concat(parts))]) as Blob;
}

describe('saved comparison bundles', () => {
  it('reads summary and per-trial scores without extracting unrelated members', async () => {
    const result = await readScoringBundle(
      archive({
        'agent-eval-results/summary.json': '{"trial_count":1}',
        'agent-eval-results/scores.jsonl': '{"task_id":"task"}\n',
        '../../ignored.html': '<script>ignored</script>',
      })
    );
    expect(result).toEqual({ summary: { trial_count: 1 }, scores: [{ task_id: 'task' }] });
  });
  it('rejects bundles without per-trial evidence', async () => {
    await expect(readScoringBundle(archive({ 'summary.json': '{}' }))).rejects.toThrow('missing');
  });
  it('rejects duplicate scorer members rather than choosing arbitrary evidence', async () => {
    await expect(
      readScoringBundle(
        archive({
          'one/summary.json': '{}',
          'two/summary.json': '{}',
          'scores.jsonl': '',
        })
      )
    ).rejects.toThrow('Ambiguous');
  });
  it('rejects malformed JSON', async () => {
    await expect(
      readScoringBundle(
        archive({
          'summary.json': '{',
          'scores.jsonl': '{}',
        })
      )
    ).rejects.toThrow();
  });
  it('rejects expanded bundles larger than the limit', async () => {
    await expect(
      readScoringBundle(
        archive({
          'large.txt': 'x'.repeat(32 * 1024 * 1024 + 1),
        })
      )
    ).rejects.toThrow('Expanded scoring bundle');
  });
  it('rejects selecting the same execution twice before making requests', async () => {
    const run = { run_id: 'same' } as TasksetRunStatus;
    await expect(loadSavedComparison('default', run, run)).rejects.toThrow('distinct');
  });
  it('rejects differing pinned tasksets before making requests', async () => {
    await expect(
      loadSavedComparison(
        'default',
        { run_id: 'before', taskset_ref: 'a' } as TasksetRunStatus,
        { run_id: 'after', taskset_ref: 'b' } as TasksetRunStatus
      )
    ).rejects.toThrow('same pinned taskset');
  });
});

vi.mock('@nemo/sdk/generated/evals/evals-plugin-agent-eval-jobs-routes', () => ({
  evalsListAgentEvaluateJobs: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/evals/evals-plugin-taskset-run-routes', () => ({
  evalsGetTasksetRunTask: vi.fn(),
  evalsListTasksetRunTasks: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/platform/jobs', () => ({ jobsGetJob: vi.fn() }));
vi.mock('@nemo/sdk/generated/fetchers/evals', () => ({ customFetch: vi.fn() }));
vi.mock('@studio/api/evaluation/agent-evaluations', () => ({
  fetchAgentEvalResult: vi.fn(),
  fetchAgentEvalBundle: vi.fn(),
}));

describe('loadSavedComparison', () => {
  const control = {
    run_id: 'control',
    taskset_ref: `default/task#${'a'.repeat(64)}`,
  } as TasksetRunStatus;
  const variant = { ...control, run_id: 'variant' };
  const skills = [`default/skill#${'b'.repeat(64)}`, `default/helper#${'c'.repeat(64)}`];
  const dates = { created_at: '2026-11-19T00:00:00Z', updated_at: '2026-11-19T00:00:00Z' };
  const summary = { task_count: 1, trial_count: 1, score_count: 2, error_count: 0 };
  const metrics = {
    scores: [
      {
        name: 'harbor_skill_metrics.package.skill.skill_structure.conforms',
        score_type: 'range' as const,
        count: 1,
        nan_count: 0,
        mean: 1,
      },
      {
        name: 'harbor_skill_metrics.tool_execution_outcome.success_rate',
        score_type: 'range' as const,
        count: 1,
        nan_count: 0,
        mean: 1,
      },
    ],
  };
  const replay = (id: string) => ({
    name: `replay-${id}`,
    status: 'completed' as const,
    spec: {
      tasks: [],
      trials: [{ id, task_id: 'task', status: 'completed' as const }],
    },
  });

  beforeEach(() => {
    vi.resetAllMocks();
    vi.mocked(evalsListTasksetRunTasks).mockImplementation(async (_workspace, id) => ({
      data: [{ task: 'task-0', task_index: 0, phase: 'succeeded', job: `job-${id}` }],
    }));
    vi.mocked(evalsGetTasksetRunTask).mockImplementation(async (_workspace, id) => ({
      task: 'task-0',
      task_index: 0,
      phase: 'succeeded',
      trials: [{ trial: id, phase: 'succeeded', reward: 1 }],
    }));
    vi.mocked(jobsGetJob).mockImplementation(async (_workspace, name) => ({
      id: name,
      name,
      workspace: 'default',
      source: 'evals.harbor-task',
      platform_spec: { steps: [] },
      fileset: 'results',
      status: 'completed',
      ...dates,
      spec: {
        spec: { target: { skills: name === 'job-variant' ? skills : [] } },
        environment: {
          agent: { image_ref: `agent@sha256:${'d'.repeat(64)}` },
          verifier: null,
        },
      },
    }));
    vi.mocked(fetchAgentEvalResult).mockImplementation(async (workspace, id) => ({
      id,
      name: id,
      workspace,
      job_id: id,
      scores: metrics,
      summary,
      bundle_ref: `default/results#${id}`,
      ...dates,
    }));
    vi.mocked(fetchAgentEvalBundle).mockImplementation(async (_workspace, ref) => {
      const id = ref?.split('#')[1] ?? '';
      return {
        tasks: [],
        trials: [
          { id, task_id: 'task', status: 'completed', measurements: { prompt_tokens: 100 } },
        ],
        scores: [
          {
            id: `score-${id}`,
            task_id: 'task',
            trial_id: id,
            metric_type: 'harbor_skill_metrics',
            status: 'completed',
            outputs: [
              { name: 'package.skill.skill_structure.conforms', value: 1 },
              { name: 'tool_execution_outcome.success_rate', value: 1 },
            ],
            diagnostics: [
              { message: 'Package finding', details: { metric: 'package.skill.skill_structure' } },
            ],
          },
        ],
      };
    });
  });

  it('uses native results for both arms with a partial replay pair and retains all variant pins', async () => {
    vi.mocked(evalsListAgentEvaluateJobs).mockResolvedValue({ data: [replay('control')] });
    const result = await loadSavedComparison('default', control, variant);
    expect(result.arms.baseline.replay?.source).toBe('run');
    expect(result.arms['with-skill']).toMatchObject({
      skill_refs: skills,
      verifier_image_ref: null,
      measurements: { prompt_tokens: 100 },
      replay: {
        source: 'run',
        summary: { scores: { scores: [metrics.scores[1]] } },
        scores: [{ outputs: [{ name: 'tool_execution_outcome.success_rate', value: 1 }] }],
      },
    });
    expect(result.package_metrics?.['package.skill.skill_structure']).toMatchObject({
      outputs: [{ name: 'conforms', value: 1 }],
      diagnostics: [{ message: 'Package finding' }],
    });
    expect(customFetch).not.toHaveBeenCalled();
  });

  it.each([
    {
      name: 'two pinned skill versions',
      controlSkills: [`default/skill#${'f'.repeat(64)}`],
      variantSkills: skills,
    },
    {
      name: 'a skill-equipped control and a skill-free variant',
      controlSkills: skills,
      variantSkills: [],
    },
  ])('loads $name', async ({ controlSkills, variantSkills }) => {
    vi.mocked(evalsListAgentEvaluateJobs).mockResolvedValue({ data: [] });
    const getJob = vi.mocked(jobsGetJob).getMockImplementation()!;
    vi.mocked(jobsGetJob).mockImplementation(async (...args) => {
      const job = await getJob(...args);
      return {
        ...job,
        spec: {
          ...job.spec,
          spec: { target: { skills: args[1] === 'job-control' ? controlSkills : variantSkills } },
        },
      };
    });
    const result = await loadSavedComparison('default', control, variant);
    expect(result.arms.baseline.skill_refs).toEqual(controlSkills);
    expect(result.arms['with-skill'].skill_refs).toEqual(variantSkills);
    expect(result.arms.baseline.replay?.source).toBe('run');
    expect(result.arms['with-skill'].replay?.source).toBe('run');
  });

  it('keeps matching scoring replays when both runs retain one', async () => {
    vi.mocked(evalsListAgentEvaluateJobs).mockResolvedValue({
      data: [replay('control'), replay('variant')],
    });
    vi.mocked(customFetch).mockResolvedValue(
      archive({
        'summary.json': JSON.stringify({ ...summary, scores: metrics }),
        'scores.jsonl': '\n',
      })
    );
    const result = await loadSavedComparison('default', control, variant);
    expect(result.arms.baseline.scoring_job?.name).toBe('replay-control');
    expect(result.arms['with-skill'].scoring_job?.name).toBe('replay-variant');
    expect(fetchAgentEvalResult).not.toHaveBeenCalled();
  });
});
