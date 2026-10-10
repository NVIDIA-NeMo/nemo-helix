// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SwitchyardFormOutput } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/formValues';
import { buildSwitchyardSpec } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/useSubmitSwitchyardOptimization';

const values = (overrides: Partial<SwitchyardFormOutput> = {}): SwitchyardFormOutput => ({
  name: 'study',
  models: ['default/big', 'default/small'],
  routingStrategies: ['random_routing'],
  strong_probability: 0.3,
  confidence_threshold: 0.6,
  base_threshold: 0.7,
  judgeModel: '',
  ...overrides,
});

describe('buildSwitchyardSpec', () => {
  it('sends only the thresholds the selected strategies read', () => {
    expect(buildSwitchyardSpec('ws', 'agent', values())).toEqual({
      strategy: 'switchyard',
      agent: 'agent',
      workspace: 'ws',
      models: ['default/big', 'default/small'],
      routing_strategies: ['random_routing'],
      strong_probability: 0.3,
    });
  });

  it('sends the classifier threshold, and the judge only when one is picked', () => {
    const withoutJudge = buildSwitchyardSpec(
      'ws',
      'agent',
      values({ routingStrategies: ['stage_router', 'llm_classifier'] })
    );
    expect(withoutJudge).toMatchObject({ confidence_threshold: 0.6, base_threshold: 0.7 });
    expect(withoutJudge).not.toHaveProperty('strong_probability');
    expect(withoutJudge).not.toHaveProperty('judge_model');

    expect(
      buildSwitchyardSpec(
        'ws',
        'agent',
        values({ routingStrategies: ['llm_classifier'], judgeModel: 'default/judge' })
      )
    ).toMatchObject({ judge_model: 'default/judge' });
  });

  it('drops the judge when the classifier is not selected', () => {
    expect(
      buildSwitchyardSpec('ws', 'agent', values({ judgeModel: 'default/judge' }))
    ).not.toHaveProperty('judge_model');
  });
});
