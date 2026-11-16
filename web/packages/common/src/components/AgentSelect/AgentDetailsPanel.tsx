// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatAgentConfigFormat } from '@nemo/common/src/components/AgentSelect/utils';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { useRelativeTimeSince } from '@nemo/common/src/components/RelativeTime';
import { type AgentConfig, getAgentModelNames } from '@nemo/common/src/utils/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema';
import { Divider, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface AgentDetailsPanelProps {
  agent: Agent;
  disabledReason?: string;
}

/** Hover card for an agent row: what it is, what it runs on, and who made it when. */
export const AgentDetailsPanel: FC<AgentDetailsPanelProps> = ({ agent, disabledReason }) => {
  const createdRelative = useRelativeTimeSince(agent.created_at ?? '');
  const updatedRelative = useRelativeTimeSince(agent.updated_at ?? '');
  const models = getAgentModelNames(agent.config as AgentConfig | undefined);
  const labels = Object.entries(agent.labels ?? {});

  return (
    <Stack className="p-3" gap="density-md" data-testid="agent-details-panel">
      <Stack gap="density-xs">
        <Text fontWeight="bold" className="break-all">
          {agent.name}
        </Text>
        {agent.description && (
          <Text className="text-secondary leading-normal" kind="body/regular/sm">
            {agent.description}
          </Text>
        )}
      </Stack>
      <Flex gap="density-sm" wrap="wrap">
        <Tag color="gray" kind="outline" readOnly>
          {formatAgentConfigFormat(agent.config_format)}
        </Tag>
        {agent.project && (
          <Tag color="gray" kind="outline" readOnly>
            {agent.project}
          </Tag>
        )}
      </Flex>
      {disabledReason && (
        <Text className="text-feedback-warning" kind="body/regular/sm">
          {disabledReason}
        </Text>
      )}
      <Divider />
      <Stack gap="density-sm">
        <KVPair
          label="Models"
          value={models.length ? models.join(', ') : undefined}
          size="narrow"
          orientation="horizontal"
        />
        {labels.length > 0 && (
          <KVPair
            label="Labels"
            value={
              <Flex gap="density-xs" wrap="wrap">
                {labels.map(([key, val]) => (
                  <Tag key={key} color="gray" kind="outline" density="compact" readOnly>
                    {`${key}=${val}`}
                  </Tag>
                ))}
              </Flex>
            }
            size="narrow"
            orientation="horizontal"
          />
        )}
        <KVPair
          label="Created by"
          value={agent.created_by ?? undefined}
          size="narrow"
          orientation="horizontal"
        />
        <KVPair
          label="Created"
          value={agent.created_at ? createdRelative : undefined}
          size="narrow"
          orientation="horizontal"
        />
        <KVPair
          label="Updated"
          value={agent.updated_at ? updatedRelative : undefined}
          size="narrow"
          orientation="horizontal"
        />
      </Stack>
    </Stack>
  );
};
