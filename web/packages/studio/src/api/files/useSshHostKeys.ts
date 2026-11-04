// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesFindGitRepositoryFiles } from '@nemo/sdk/generated/platform/files';
import type { SshHostKey } from '@nemo/sdk/generated/platform/schema';
import { AGENT_CONFIG_FILENAME } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import {
  gitStorageConfig,
  sshHostOf,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/git';
import { type UseQueryResult, useQuery } from '@tanstack/react-query';

export interface SshHostKeysLookup {
  scan: UseQueryResult<SshHostKey[], unknown>;
  /** The scan has answered for the URL passed in now. */
  settled: boolean;
}

/** The keys an SSH remote's host presents: find-files without known_hosts reads nothing and returns them. */
export const useSshHostKeys = (
  workspace: string,
  url: string | undefined,
  sshKeySecret: string | undefined
): SshHostKeysLookup => {
  const host = url ? sshHostOf(url) : undefined;
  const scan = useQuery({
    queryKey: ['ssh-host-keys', workspace, host, url, sshKeySecret],
    queryFn: async ({ signal }) => {
      const { host_keys } = await filesFindGitRepositoryFiles(
        workspace,
        {
          storage: gitStorageConfig({ url: url ?? '', path: '' }, sshKeySecret ?? '', ''),
          file_name: AGENT_CONFIG_FILENAME,
        },
        signal
      );
      return host_keys ?? [];
    },
    enabled: Boolean(workspace && host && sshKeySecret),
    staleTime: Infinity,
    retry: false,
  });
  return { scan, settled: !scan.isFetching };
};
