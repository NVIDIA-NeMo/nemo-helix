// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesScanSshHostKeys } from '@nemo/sdk/generated/platform/files';
import type { ScanSshHostKeysResponse } from '@nemo/sdk/generated/platform/schema';
import { sshHostOf } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/git';
import { type UseQueryResult, useQuery } from '@tanstack/react-query';

export interface SshHostKeysLookup {
  scan: UseQueryResult<ScanSshHostKeysResponse, unknown>;
  /** The scan has answered for the URL passed in now. */
  settled: boolean;
}

/** The keys an SSH remote's host presents, looked up once per host. */
export const useSshHostKeys = (workspace: string, url: string | undefined): SshHostKeysLookup => {
  const host = url ? sshHostOf(url) : undefined;
  const scan = useQuery({
    queryKey: ['ssh-host-keys', workspace, host],
    queryFn: ({ signal }) => filesScanSshHostKeys(workspace, { url: url ?? '' }, signal),
    enabled: Boolean(workspace && host),
    staleTime: Infinity,
    retry: false,
  });
  return { scan, settled: !scan.isFetching };
};
