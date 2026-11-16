// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentRow } from '@nemo/common/src/components/AgentSelect/types';
import { FABRIC_CONFIG_FORMAT } from '@nemo/common/src/utils/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema';

const FORMAT_LABELS: Record<string, string> = {
  [FABRIC_CONFIG_FORMAT]: 'Fabric',
  'nat-workflow-v1': 'NAT workflow',
};

/** Display name for an agent's `config_format`; an unset format is the legacy NAT default. */
export const formatAgentConfigFormat = (format: string | undefined): string =>
  FORMAT_LABELS[format ?? 'nat-workflow-v1'] ?? format ?? '';

export const groupAgentsByFormat = (agent: Agent): string =>
  formatAgentConfigFormat(agent.config_format);

/** Client-side match for when the caller does not search server-side. */
export const matchesAgentSearch = (agent: Agent, search: string): boolean => {
  const term = search.trim().toLowerCase();
  return (
    !term || [agent.name, agent.description].some((field) => field?.toLowerCase().includes(term))
  );
};

/** Flattens agents into virtualizer rows, with a heading per group when `groupBy` is given. */
export const buildAgentRows = (
  agents: readonly Agent[],
  groupBy?: (agent: Agent) => string
): AgentRow[] => {
  const toRow = (agent: Agent): AgentRow => ({
    kind: 'agent',
    key: agent.id ?? agent.name ?? '',
    agent,
  });
  if (!groupBy) return agents.map(toRow);

  const groups = new Map<string, Agent[]>();
  for (const agent of agents) {
    const label = groupBy(agent);
    groups.set(label, [...(groups.get(label) ?? []), agent]);
  }
  return [...groups].flatMap(([label, members]) => [
    { kind: 'heading' as const, key: `heading-${label}`, label },
    ...members.map(toRow),
  ]);
};
