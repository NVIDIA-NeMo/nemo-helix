// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { ChatSidePane } from '@studio/routes/agents/AgentDetailRoute/ChatSidePane';
import { shortRevision } from '@studio/routes/agents/AgentDetailRoute/helpers';
import type { FC, ReactNode } from 'react';

const Field: FC<{ label: string; children: ReactNode }> = ({ label, children }) => (
  <Stack gap="density-xs">
    <Text kind="label/regular/sm" className="text-secondary">
      {label}
    </Text>
    {children}
  </Stack>
);

const Value: FC<{ children: ReactNode; mono?: boolean }> = ({ children, mono }) => (
  <Text kind="body/regular/sm" className={mono ? 'font-mono break-all' : 'break-words'}>
    {children}
  </Text>
);

interface AgentRuntimePanelProps {
  deployment?: AgentDeployment;
  modelNames: string[];
  className?: string;
}

/** Summarizes the deployment the chat talks to: its status, how it runs, and its models. */
export const AgentRuntimePanel: FC<AgentRuntimePanelProps> = ({
  deployment,
  modelNames,
  className,
}) => (
  <ChatSidePane title="Runtime" side="start" className={className}>
    {deployment ? (
      <>
        <Field label="Deployment">
          <Value mono>{deployment.name}</Value>
          <Flex>
            <StatusBadge status={deployment.status} />
          </Flex>
        </Field>
        {deployment.deployment_mode ? (
          <Field label="Mode">
            <Value>{deployment.deployment_mode}</Value>
          </Field>
        ) : null}
        {deployment.spec_revision ? (
          <Field label="Revision">
            <Value mono>{shortRevision(deployment.spec_revision)}</Value>
          </Field>
        ) : null}
        {deployment.error ? (
          <Field label="Error">
            <Text kind="body/regular/sm" className="text-feedback-danger [overflow-wrap:anywhere]">
              {deployment.error}
            </Text>
          </Field>
        ) : null}
      </>
    ) : (
      <Text kind="body/regular/sm" className="text-secondary">
        No deployment selected.
      </Text>
    )}
    {modelNames.length > 0 ? (
      <Field label={modelNames.length === 1 ? 'Model' : 'Models'}>
        {modelNames.map((model) => (
          <Value key={model} mono>
            {model}
          </Value>
        ))}
      </Field>
    ) : null}
  </ChatSidePane>
);
