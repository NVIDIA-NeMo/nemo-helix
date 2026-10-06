// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesFindGitRepositoryFiles } from '@nemo/sdk/generated/platform/files';
import { AGENT_CONFIG_FILENAME } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import { gitStorageConfig } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/git';
import { agentDirectoryOf } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { type UseQueryResult, useQuery } from '@tanstack/react-query';

/** Everything a lookup depends on. The directory is left out: the whole repository is searched. */
export interface GitAgentLookup {
  url: string;
  ref?: string;
  sshKeySecret: string;
  knownHosts: string;
}

export interface GitAgentDirectories {
  /** Repository paths of directories holding an agent.yaml; the root is `''`. */
  search: UseQueryResult<string[], unknown>;
  /** The search has answered for the lookup passed in now. */
  settled: boolean;
}

export const useGitAgentDirectories = (
  workspace: string,
  lookup: GitAgentLookup | undefined
): GitAgentDirectories => {
  const search = useQuery({
    queryKey: ['git-agent-directories', workspace, lookup],
    queryFn: async ({ signal }) => {
      if (!lookup) return [];
      const { url, ref, sshKeySecret, knownHosts } = lookup;
      const { paths } = await filesFindGitRepositoryFiles(
        workspace,
        {
          storage: gitStorageConfig({ url, ref, path: '' }, sshKeySecret, knownHosts),
          file_name: AGENT_CONFIG_FILENAME,
        },
        signal
      );
      return paths
        .map(agentDirectoryOf)
        .filter((directory): directory is string => directory !== undefined);
    },
    enabled: Boolean(workspace && lookup),
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
  return { search, settled: !search.isFetching };
};
