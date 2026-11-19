// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type {
  ComparisonArm,
  ComparisonReport,
} from '@studio/routes/evaluation/SkillComparisonRoute/report';

export function makeSkillComparisonReport(): ComparisonReport {
  const arm = (treatment: boolean): ComparisonArm => ({
    run_id: treatment ? 'run-0000000000000002' : 'run-0000000000000001',
    job_name: treatment ? 'skill-task' : 'baseline-task',
    reward: 1,
    agent_image_ref: `registry.example/task@sha256:${'a'.repeat(64)}`,
    verifier_image_ref: `registry.example/verifier@sha256:${'b'.repeat(64)}`,
    bundle: `${treatment ? 'with-skill' : 'baseline'}/agent-eval-results`,
    discovery: treatment,
    activation: treatment ? true : null,
    measurements: {
      prompt_tokens: 1000,
      cache_read_tokens: 600,
      completion_tokens: 100,
      total_tokens: 1100,
    },
    scoring_job: { name: treatment ? 'skill-scoring' : 'baseline-scoring' },
    replay: {
      bundle: `${treatment ? 'with-skill' : 'baseline'}-scoring-results.tar.gz`,
      summary: {
        trial_count: 1,
        error_count: 0,
        scores: {
          scores: [
            { name: 'harbor_reward.reward', score_type: 'range', mean: 1, count: 1, nan_count: 0 },
            {
              name: 'skill_activation.activated',
              score_type: 'range',
              mean: treatment ? 1 : 0,
              count: treatment ? 1 : 0,
              nan_count: treatment ? 0 : 1,
            },
            {
              name: 'token_efficiency.efficiency_tokens',
              score_type: 'range',
              mean: 500,
              count: 1,
              nan_count: 0,
            },
            {
              name: 'llm-judge.assertion_1',
              score_type: 'range',
              mean: treatment ? 1 : 0,
              count: 1,
              nan_count: 0,
            },
            {
              name: 'view.expected_behavior',
              score_type: 'range',
              mean: treatment ? 1 : 0,
              count: 1,
              nan_count: 0,
            },
          ],
        },
        metric_coverage: {
          harbor_reward: { reward: { total: 1, scored: 1, failed: 0, missing: 0 } },
          skill_activation: {
            activated: {
              total: 1,
              scored: treatment ? 1 : 0,
              failed: 0,
              missing: treatment ? 0 : 1,
            },
          },
        },
      },
      scores: [
        {
          task_id: 'exact-match-task',
          trial_id: treatment ? 'trial-skill' : 'trial-baseline',
          metric_type: 'harbor_reward',
          status: 'completed',
          outputs: [{ name: 'reward', value: 1 }],
          diagnostics: [],
        },
        {
          task_id: 'exact-match-task',
          trial_id: treatment ? 'trial-skill' : 'trial-baseline',
          metric_type: 'llm-judge',
          status: 'completed',
          outputs: [{ name: 'assertion_1', value: treatment ? 1 : 0 }],
          diagnostics: [{ message: 'Fixture assertion evidence.' }],
        },
      ],
    },
  });
  return {
    workspace: DEFAULT_WORKSPACE,
    context: 'test-context',
    taskset_ref: `${DEFAULT_WORKSPACE}/exact-match#${'c'.repeat(64)}`,
    skill_ref: `${DEFAULT_WORKSPACE}/nemo-evals-plugin#${'d'.repeat(64)}`,
    judge_model_ref: `${DEFAULT_WORKSPACE}/judge`,
    package_metrics: {
      skill_structure: { outputs: [{ name: 'defects', value: 0 }], diagnostics: [] },
    },
    arms: { baseline: arm(false), 'with-skill': arm(true) },
  };
}
