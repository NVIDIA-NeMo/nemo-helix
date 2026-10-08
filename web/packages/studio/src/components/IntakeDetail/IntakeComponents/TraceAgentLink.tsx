// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import { Anchor } from '@nvidia/foundations-react-core';
import { AGENTS_ENABLED } from '@studio/constants/environment';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { EMPTY_VALUE } from '@studio/util/intakeTelemetry';
import type { FC } from 'react';
import { Link } from 'react-router';

interface TraceAgentLinkProps {
  workspace: string;
  agentName?: string;
}

/**
 * A trace's `agent_name`, linked to the agent page only when an agent by that name is registered
 * in the workspace. Traces can name agents that were never registered, so a 404 renders plain text
 * and must not retry.
 */
export const TraceAgentLink: FC<TraceAgentLinkProps> = ({ workspace, agentName }) => {
  const { isSuccess } = useAgentsGetAgent(workspace, agentName ?? '', {
    query: { enabled: AGENTS_ENABLED && !!agentName, retry: false },
  });

  if (!agentName) return <>{EMPTY_VALUE}</>;
  if (!isSuccess) return <>{agentName}</>;

  return (
    <Anchor asChild>
      <Link to={getAgentDetailRoute(workspace, agentName)} className="truncate" title={agentName}>
        {agentName}
      </Link>
    </Anchor>
  );
};
