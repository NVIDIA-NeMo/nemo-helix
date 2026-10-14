// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  parseSwitchyardIndex,
  routeThreshold,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/switchyardResults';

const LOCATION = { workspace: 'default', fileset: 'job-results' };

const INDEX = {
  strategy: 'switchyard',
  agent: 'default/support-bot',
  workspace: 'default',
  combinations: [
    {
      virtual_model: 'support-bot-1a2b3c4d',
      config_type: 'random_routing',
      capable: 'default/frontier',
      efficient: 'default/nano',
      judge: null,
      config: {
        strong: { model: 'default/frontier' },
        weak: { model: 'default/nano' },
        strong_probability: 0.8,
      },
      agent_config: 'agent-support-bot-1a2b3c4d.yaml',
    },
    {
      virtual_model: 'support-bot-5e6f7a8b',
      config_type: 'llm_classifier',
      capable: 'default/frontier',
      efficient: 'default/nano',
      judge: 'default/frontier',
      config: { mode: 'capability', base_threshold: 0.4 },
    },
    { virtual_model: 'missing-models' },
  ],
};

describe('parseSwitchyardIndex', () => {
  it('reads each route and resolves its agent config beside the index', () => {
    const results = parseSwitchyardIndex(
      JSON.stringify(INDEX),
      'switchyard/switchyard-result.json',
      LOCATION,
      'fallback'
    );

    expect(results.agent).toBe('default/support-bot');
    expect(results.workspace).toBe('default');
    expect(results.location).toBe(LOCATION);
    expect(results.routes).toHaveLength(2);
    expect(results.routes[0]).toMatchObject({
      virtualModel: 'support-bot-1a2b3c4d',
      configType: 'random_routing',
      judge: null,
      agentConfigPath: 'switchyard/agent-support-bot-1a2b3c4d.yaml',
    });
    expect(results.routes[1]).toMatchObject({
      judge: 'default/frontier',
      agentConfigPath: undefined,
    });
  });

  it('falls back to the job workspace and a root-level path', () => {
    const results = parseSwitchyardIndex(
      JSON.stringify({ combinations: [INDEX.combinations[0]] }),
      'switchyard-result.json',
      LOCATION,
      'fallback'
    );

    expect(results.workspace).toBe('fallback');
    expect(results.routes[0]?.agentConfigPath).toBe('agent-support-bot-1a2b3c4d.yaml');
  });

  it('rejects an index that is not a JSON object', () => {
    expect(() => parseSwitchyardIndex('[]', 'switchyard-result.json', LOCATION, 'w')).toThrow(
      /not a JSON object/
    );
  });
});

describe('routeThreshold', () => {
  it("reads each strategy's own knob", () => {
    const [random, classifier] = parseSwitchyardIndex(
      JSON.stringify(INDEX),
      'switchyard-result.json',
      LOCATION,
      'w'
    ).routes;

    expect(routeThreshold(random!)).toBe(0.8);
    expect(routeThreshold(classifier!)).toBe(0.4);
  });
});
