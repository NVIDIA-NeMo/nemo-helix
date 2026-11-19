// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ArmName } from '@studio/routes/evaluation/SkillComparisonRoute/report';

export const ARM_LABELS: Record<ArmName, string> = { baseline: 'Control', 'with-skill': 'Variant' };

const scanScores = ['complete', 'clean'];
const readinessScores = ['score'];
const normalizedOutputs: Record<string, string[]> = {
  skill_structure: ['conforms'],
  skill_reference_integrity: [],
  skill_context_efficiency: ['within_budget'],
  skill_activation: ['activated', 'as_expected'],
  token_efficiency: ['score'],
  skill_static_correctness: readinessScores,
  skill_description_quality: readinessScores,
  skill_reliability_readiness: readinessScores,
  skill_script_quality: [],
  skill_unicode_safety: ['clean'],
  skill_security_scan: scanScores,
  skill_sensitive_data: scanScores,
  skill_code_risk: scanScores,
  skill_license: scanScores,
  skill_dependency_compliance: scanScores,
  skill_package_configuration_risk: scanScores,
  agent_safety: scanScores,
  skill_bundled_test: ['complete', 'pass_rate', 'coverage', 'meets_coverage'],
  skill_overactivation: ['overactivated', 'as_expected'],
  skill_routing: ['complete', 'as_expected'],
  expected_script_invocation: ['complete', 'invoked'],
  workflow_compliance: ['complete', 'score'],
  error_recovery: ['complete', 'recovery_rate'],
  tool_productivity: ['complete', 'score'],
  tool_selection: ['complete', 'precision', 'recall', 'f1'],
  tool_argument_compliance: ['complete', 'score'],
  tool_execution_outcome: ['complete', 'success_rate'],
  component_coordination: ['complete', 'score'],
  component_usage: ['complete', 'coverage', 'multi_component'],
};

export function isNormalizedOutput(name: string): boolean {
  const [metric, output] = name.split('.', 2);
  return !Object.hasOwn(normalizedOutputs, metric) || normalizedOutputs[metric].includes(output);
}

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
  if (Object.hasOwn(metricLabels, name)) return metricLabels[name];
  const judge = /^llm-judge\.assertion_(\d+)(\.label)?$/.exec(name);
  if (judge) return `Judge assertion ${judge[1]}${judge[2] ? ' label' : ''}`;
  const pass = /^(.*)\.pass@(\d+)$/.exec(name);
  if (pass) return `${metricLabel(pass[1])} · pass@${pass[2]}`;
  return name
    .replace(/^view\./, '')
    .replace(/[_.-]+/g, ' ')
    .replace(/^./, (letter) => letter.toUpperCase());
}
