// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { ConfigValue } from '@studio/routes/agents/AgentDetailRoute/ConfigValue';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import type { FC } from 'react';

/** Config keys rendered by dedicated structured panels below. */
const STRUCTURED_KEYS = ['workflow', 'llms', 'functions'] as const;

interface AgentConfigPanelsProps {
  agent?: Agent;
}

/**
 * Every field of the agent's config, grouped into structured panels with an
 * "Additional configuration" fallback so nothing in the spec is hidden.
 */
export const AgentConfigPanels: FC<AgentConfigPanelsProps> = ({ agent }) => {
  const config = (agent?.config ?? {}) as Record<string, unknown>;

  const workflow = asRecord(config['workflow']);
  const llms = asRecord(config['llms']);
  const functions = asRecord(config['functions']);
  const extraEntries = Object.entries(config).filter(
    ([key]) => !STRUCTURED_KEYS.includes(key as (typeof STRUCTURED_KEYS)[number])
  );

  return (
    <>
      {workflow && (
        <DetailPanel title="Workflow">
          <ConfigEntries data={workflow} />
        </DetailPanel>
      )}

      {llms && (
        <DetailPanel title="Models">
          <ConfigEntries data={llms} />
        </DetailPanel>
      )}

      {functions && (
        <DetailPanel title="Tools">
          <ConfigEntries data={functions} />
        </DetailPanel>
      )}

      {extraEntries.length > 0 && (
        <DetailPanel title="Additional configuration" defaultCollapsed>
          <Stack gap="2">
            {extraEntries.map(([key, value]) => (
              <ConfigValue key={key} label={key} value={value} />
            ))}
          </Stack>
        </DetailPanel>
      )}

      {agent && Object.keys(config).length === 0 && (
        <DetailPanel title="Configuration">
          <Text kind="body/regular/sm" className="text-secondary">
            This agent has no stored configuration.
          </Text>
        </DetailPanel>
      )}
    </>
  );
};

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;

const ConfigEntries: FC<{ data: Record<string, unknown> }> = ({ data }) => (
  <Stack gap="2">
    {Object.entries(data).map(([key, value]) => (
      <ConfigValue key={key} label={key} value={value} />
    ))}
  </Stack>
);
