// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelProvider } from '@nemo/sdk/generated/platform/schema';
import type { SwitchyardRoute } from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/switchyardResults';

/** Which side of a route served a response; `unknown` when its model matches neither. */
export type RouteRole = 'capable' | 'efficient' | 'unknown';

export interface RoutedResponse {
  id: string;
  /** The `model` the gateway's response reported, if it carried one. */
  responseModel?: string;
}

export interface RoleTally {
  role: RouteRole;
  /** The model entity behind the role; the raw response model names for `unknown`. */
  models: string[];
  count: number;
  share: number;
}

/**
 * Every served model name each model entity goes by, across providers. The gateway sends the
 * served name upstream, and the upstream reports it back as the response's `model`.
 */
export const servedNamesByEntity = (providers: ModelProvider[]): Map<string, Set<string>> => {
  const names = new Map<string, Set<string>>();
  for (const provider of providers) {
    for (const served of provider.served_models ?? []) {
      if (!served.model_entity_id || !served.served_model_name) continue;
      const set = names.get(served.model_entity_id) ?? new Set<string>();
      set.add(served.served_model_name);
      names.set(served.model_entity_id, set);
    }
  }
  return names;
};

/** The names a response could report for a model entity: its ref, bare name, and served names. */
const aliasesFor = (entityRef: string, servedNames: Map<string, Set<string>>): Set<string> => {
  const aliases = new Set<string>([entityRef, entityRef.split('/').pop() ?? entityRef]);
  for (const served of servedNames.get(entityRef) ?? []) aliases.add(served);
  return new Set([...aliases].map((alias) => alias.toLowerCase()));
};

export const classifyResponseModel = (
  route: Pick<SwitchyardRoute, 'capable' | 'efficient'>,
  responseModel: string | undefined,
  servedNames: Map<string, Set<string>>
): RouteRole => {
  if (!responseModel) return 'unknown';
  const name = responseModel.toLowerCase();
  if (aliasesFor(route.capable, servedNames).has(name)) return 'capable';
  if (aliasesFor(route.efficient, servedNames).has(name)) return 'efficient';
  return 'unknown';
};

/**
 * How a route's responses split between its capable and efficient model. Both sides are always
 * listed, so an unused side shows as zero rather than disappearing; `unknown` only when present.
 */
export const tallyRoutes = (
  route: Pick<SwitchyardRoute, 'capable' | 'efficient'>,
  responses: readonly RoutedResponse[],
  servedNames: Map<string, Set<string>>
): RoleTally[] => {
  const counts: Record<RouteRole, number> = { capable: 0, efficient: 0, unknown: 0 };
  const unknownModels = new Set<string>();
  for (const response of responses) {
    const role = classifyResponseModel(route, response.responseModel, servedNames);
    counts[role] += 1;
    if (role === 'unknown') unknownModels.add(response.responseModel ?? 'No model reported');
  }
  const total = responses.length;
  const share = (count: number) => (total ? count / total : 0);
  const tallies: RoleTally[] = [
    {
      role: 'capable',
      models: [route.capable],
      count: counts.capable,
      share: share(counts.capable),
    },
    {
      role: 'efficient',
      models: [route.efficient],
      count: counts.efficient,
      share: share(counts.efficient),
    },
  ];
  if (counts.unknown) {
    tallies.push({
      role: 'unknown',
      models: [...unknownModels],
      count: counts.unknown,
      share: share(counts.unknown),
    });
  }
  return tallies;
};
