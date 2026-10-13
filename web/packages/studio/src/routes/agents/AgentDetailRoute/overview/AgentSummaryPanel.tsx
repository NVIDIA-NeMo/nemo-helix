// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import type { FC } from 'react';

interface AgentSummaryPanelProps {
  agent?: Agent;
  modelNames: string[];
}

/** At-a-glance identity for the agent, alongside the overview statistics. */
export const AgentSummaryPanel: FC<AgentSummaryPanelProps> = ({ agent, modelNames }) => (
  <DetailPanel title="Details">
    <Stack gap="3">
      {agent?.config_format && (
        <KVPair label="Type" value={agent.config_format} size="narrow" orientation="horizontal" />
      )}
      {agent?.project && (
        <KVPair label="Project" value={agent.project} size="narrow" orientation="horizontal" />
      )}
      <KVPair
        label={modelNames.length > 1 ? 'Models' : 'Model'}
        size="narrow"
        orientation="horizontal"
        value={
          modelNames.length > 0 ? (
            <Stack gap="1">
              {modelNames.map((model) => (
                <Text key={model} kind="body/semibold/md">
                  {model}
                </Text>
              ))}
            </Stack>
          ) : null
        }
      />
      <KVPair
        label="Agent ID"
        value={agent?.id}
        size="narrow"
        orientation="horizontal"
        truncate
        loading={!agent}
      />
      <KVPair
        label="Description"
        value={agent?.description}
        size="narrow"
        orientation="horizontal"
        loading={!agent}
      />
      <KVPair
        label="Created"
        size="narrow"
        orientation="horizontal"
        value={agent?.created_at ? <RelativeTime datetime={agent.created_at} /> : null}
        loading={!agent}
      />
      <KVPair
        label="Updated"
        size="narrow"
        orientation="horizontal"
        value={agent?.updated_at ? <RelativeTime datetime={agent.updated_at} /> : null}
        loading={!agent}
      />
    </Stack>
  </DetailPanel>
);
