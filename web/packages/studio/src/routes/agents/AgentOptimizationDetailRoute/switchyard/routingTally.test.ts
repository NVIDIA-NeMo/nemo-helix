// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelProvider } from '@nemo/sdk/generated/platform/schema';
import {
  classifyResponseModel,
  type RoutedResponse,
  servedNamesByEntity,
  tallyRoutes,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/routingTally';

const ROUTE = { capable: 'default/frontier', efficient: 'default/nemotron-nano' };

const provider = (served: { model_entity_id: string; served_model_name: string }[]) =>
  ({ name: 'build', workspace: 'default', served_models: served }) as unknown as ModelProvider;

const SERVED = servedNamesByEntity([
  provider([
    { model_entity_id: 'default/frontier', served_model_name: 'meta/llama-3.3-70b-instruct' },
    { model_entity_id: 'default/nemotron-nano', served_model_name: 'nvidia/nemotron-nano-9b-v2' },
  ]),
]);

const response = (responseModel: string | undefined, index = 0): RoutedResponse => ({
  id: `r-${index}`,
  responseModel,
});

describe('classifyResponseModel', () => {
  it.each([
    ['meta/llama-3.3-70b-instruct', 'capable'],
    ['NVIDIA/Nemotron-Nano-9B-v2', 'efficient'],
    ['default/frontier', 'capable'],
    ['nemotron-nano', 'efficient'],
    ['some-other-model', 'unknown'],
    [undefined, 'unknown'],
  ] as const)('classifies %s as %s', (model, role) => {
    expect(classifyResponseModel(ROUTE, model, SERVED)).toBe(role);
  });
});

describe('tallyRoutes', () => {
  it('counts each side and keeps an unused side at zero', () => {
    const responses = [
      response('nvidia/nemotron-nano-9b-v2', 1),
      response('nvidia/nemotron-nano-9b-v2', 2),
      response('nvidia/nemotron-nano-9b-v2', 3),
    ];

    expect(tallyRoutes(ROUTE, responses, SERVED)).toEqual([
      { role: 'capable', models: ['default/frontier'], count: 0, share: 0 },
      { role: 'efficient', models: ['default/nemotron-nano'], count: 3, share: 1 },
    ]);
  });

  it('lists responses matching neither model under unknown, by the name they reported', () => {
    const tallies = tallyRoutes(
      ROUTE,
      [response('meta/llama-3.3-70b-instruct', 1), response('mystery', 2), response(undefined, 3)],
      SERVED
    );

    expect(tallies.map((tally) => [tally.role, tally.count])).toEqual([
      ['capable', 1],
      ['efficient', 0],
      ['unknown', 2],
    ]);
    expect(tallies[2]?.models).toEqual(['mystery', 'No model reported']);
    expect(tallies[0]?.share).toBeCloseTo(1 / 3);
  });
});
