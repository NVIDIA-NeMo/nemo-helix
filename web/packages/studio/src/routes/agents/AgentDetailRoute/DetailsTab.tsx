// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { MarkdownContent } from '@nemo/common/src/components/MarkdownContent';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { isDefined } from '@nemo/common/src/utils/list';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { useAgentSpecFileset } from '@studio/api/agents/useAgentSpecFileset';
import type { AgentConfig } from '@studio/components/dataViews/AgentsDataView';
import { getAgentModelNames } from '@studio/components/dataViews/AgentsDataView/utils';
import { ConfigValue } from '@studio/routes/agents/AgentDetailRoute/ConfigValue';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import { SourcePanel } from '@studio/routes/agents/AgentDetailRoute/SourcePanel';
import { getFilesetRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Link } from 'react-router';

/** Config keys rendered by the Summary panel or dedicated structured panels below. */
const RENDERED_KEYS = new Set([
  'name',
  'workspace',
  'project',
  'description',
  'config_format',
  'workflow',
  'llms',
  'functions',
]);

interface DetailsTabProps {
  workspace: string;
  agentName?: string;
  agent?: Agent;
}

/**
 * Read-only view of the exact spec an agent is built with: identity metadata
 * plus every field of its config, grouped into structured panels with an
 * "Additional configuration" fallback so nothing in the spec is hidden.
 */
export const DetailsTab: FC<DetailsTabProps> = ({ workspace, agentName, agent }) => {
  const config = (agent?.config ?? {}) as Record<string, unknown>;
  const models = getAgentModelNames(agent?.config as AgentConfig | undefined);
  const { data: fileset } = useAgentSpecFileset(workspace, agent?.name ?? agentName);

  const workflow = asRecord(config['workflow']);
  const llms = asRecord(config['llms']);
  const functions = asRecord(config['functions']);
  const extraEntries = Object.entries(config).filter(([key]) => !RENDERED_KEYS.has(key));

  return (
    <Stack gap="4" className="mx-auto w-full max-w-3xl pb-6">
      <DetailPanel title="Summary">
        <Stack gap="2">
          <KVPair label="Name" value={agent?.name ?? agentName} />
          <KVPair label="Workspace" value={agent?.workspace ?? workspace} />
          {isDefined(agent?.project) && agent.project && (
            <KVPair label="Project" value={agent.project} />
          )}
          {isDefined(agent?.description) && agent.description && (
            <Stack gap="1">
              <Text kind="label/regular/sm" className="text-secondary">
                Description
              </Text>
              <MarkdownContent content={agent.description} disableImages />
            </Stack>
          )}
          {models.length > 0 && <KVPair label="Model" value={models.join(', ')} />}
          {isDefined(agent?.config_format) && (
            <KVPair label="Config format" value={agent.config_format} />
          )}
          {fileset && (
            <KVPair
              label="Fileset"
              value={
                <Link
                  to={getFilesetRoute(fileset.workspace, fileset.name)}
                  className="text-primary underline"
                >
                  {fileset.name}
                </Link>
              }
            />
          )}
          {isDefined(agent?.id) && <KVPair label="Agent ID" value={agent.id} truncate />}
          {isDefined(agent?.created_at) && agent.created_at && (
            <KVPair label="Created" value={<RelativeTime datetime={agent.created_at} />} />
          )}
          {isDefined(agent?.updated_at) && agent.updated_at && (
            <KVPair label="Updated" value={<RelativeTime datetime={agent.updated_at} />} />
          )}
        </Stack>
      </DetailPanel>

      <SourcePanel workspace={workspace} agentName={agent?.name ?? agentName} />

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

      {Object.keys(config).length === 0 && (
        <DetailPanel title="Configuration">
          <Text kind="body/regular/sm" className="text-secondary">
            This agent has no stored configuration.
          </Text>
        </DetailPanel>
      )}
    </Stack>
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
