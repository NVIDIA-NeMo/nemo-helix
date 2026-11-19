// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ArmName } from '@studio/routes/evaluation/SkillComparisonRoute/report';

export const ARM_LABELS: Record<ArmName, string> = { baseline: 'Control', 'with-skill': 'Variant' };

const metricLabels: Record<string, string> = {
  'harbor_reward.reward': 'Task success',
  'skill_activation.activated': 'Skill activation',
  'skill_activation.as_expected': 'Expected skill use',
  'token_efficiency.score': 'Token efficiency score (0–1)',
  'token_efficiency.efficiency_tokens': 'Mean uncached prompt + completion (tokens)',
  'view.expected_behavior': 'Expected behavior',
  'view.all_assertions_met': 'All assertions met',
};

export function metricLabel(name: string): string {
  if (metricLabels[name]) return metricLabels[name];
  const judge = /^llm-judge\.assertion_(\d+)(\.label)?$/.exec(name);
  if (judge) return `Judge assertion ${judge[1]}${judge[2] ? ' label' : ''}`;
  const pass = /^(.*)\.pass@(\d+)$/.exec(name);
  if (pass) return `${metricLabel(pass[1])} · pass@${pass[2]}`;
  return name
    .replace(/^view\./, '')
    .replace(/[_.-]+/g, ' ')
    .replace(/^./, (letter) => letter.toUpperCase());
}
