// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Block, Stack, Text } from '@nvidia/foundations-react-core';
import { ChatInfoSection } from '@studio/routes/agents/AgentDetailRoute/ChatInfoSection';
import { shortRevision } from '@studio/routes/agents/AgentDetailRoute/helpers';
import type { FC, ReactNode } from 'react';

const Row: FC<{ label: string; value: ReactNode }> = ({ label, value }) => (
  <Block className="py-density-sm">
    <KVPair label={label} value={value} size="narrow" orientation="horizontal" />
  </Block>
);

interface AgentRuntimePanelProps {
  /** The header's selection; its name and status already show in the header. */
  deployment?: AgentDeployment;
  modelNames: string[];
}

/** How the deployment the chat talks to runs, and the models behind it. */
export const AgentRuntimePanel: FC<AgentRuntimePanelProps> = ({ deployment, modelNames }) => (
  <ChatInfoSection title="Details">
    <Stack className="divide-y divide-base">
      <Row
        label={modelNames.length > 1 ? 'Models' : 'Model'}
        value={
          modelNames.length > 0 ? (
            <Stack gap="1">
              {modelNames.map((model) => (
                <span key={model} className="break-all">
                  {model}
                </span>
              ))}
            </Stack>
          ) : null
        }
      />
      {deployment?.deployment_mode ? <Row label="Mode" value={deployment.deployment_mode} /> : null}
      {deployment?.spec_revision ? (
        <Row
          label="Revision"
          value={<span className="font-mono">{shortRevision(deployment.spec_revision)}</span>}
        />
      ) : null}
      {deployment?.error ? (
        <Row
          label="Error"
          value={
            <Text kind="body/regular/sm" className="text-feedback-danger [overflow-wrap:anywhere]">
              {deployment.error}
            </Text>
          }
        />
      ) : null}
    </Stack>
  </ChatInfoSection>
);
