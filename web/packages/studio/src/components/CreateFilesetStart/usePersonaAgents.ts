// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fetchAgentsForSelect } from '@studio/components/agents/AgentBlockingInput/agentQueries';
import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';

/** The workspace's named agents, shared by the Personas tile and its panel. */
export const usePersonaAgents = (workspace: string) => {
  const query = useQuery({
    queryKey: ['persona-test-set', 'agents', workspace],
    queryFn: ({ signal }) => fetchAgentsForSelect(workspace, signal),
    enabled: !!workspace,
  });
  const agents = useMemo(() => (query.data ?? []).filter((agent) => !!agent.name), [query.data]);
  return { ...query, agents };
};
