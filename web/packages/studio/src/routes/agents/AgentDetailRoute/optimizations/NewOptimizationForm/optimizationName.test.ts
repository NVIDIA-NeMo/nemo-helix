// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_NAME_MAX_LENGTH, ENTITY_NAME_REGEXP } from '@nemo/common/src/utils/entityName';
import { buildOptimizationName } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizationName';

const now = new Date(2026, 8, 10, 9, 4, 7);

describe('buildOptimizationName', () => {
  it('names the agent, the intent, and when the run started', () => {
    expect(buildOptimizationName('email-triage', 'accuracy', now)).toBe(
      'email-triage-accuracy-0910-090407'
    );
  });

  it('carries the intent so two studies on one agent read apart', () => {
    expect(buildOptimizationName('email-triage', 'cost', now)).toBe(
      'email-triage-cost-0910-090407'
    );
  });

  it('truncates the agent rather than the suffix on a long name', () => {
    const name = buildOptimizationName('a'.repeat(80), 'creativity', now);

    expect(name.length).toBeLessThanOrEqual(ENTITY_NAME_MAX_LENGTH);
    expect(name).toMatch(/-creativity-0910-090407$/);
  });

  it('sanitizes an agent name that is not itself entity-safe', () => {
    const name = buildOptimizationName('Email Triage!', 'brevity', now);

    expect(name).toBe('email-triage-brevity-0910-090407');
    expect(ENTITY_NAME_REGEXP.test(name)).toBe(true);
  });
});
