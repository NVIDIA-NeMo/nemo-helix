// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import {
  filesRefreshFileset,
  getFilesRetrieveFilesetQueryKey,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput, GithubStorageConfig } from '@nemo/sdk/generated/platform/schema';
import { agentSpecFilesetName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { useMutation, useQueryClient } from '@tanstack/react-query';

export interface AgentSpecSource {
  owner: string;
  repo: string;
  /** `owner/repo`, with the sub-directory appended when the fileset is scoped to one. */
  repository: string;
  /** The mutable ref the fileset tracks, if any. Absent when it was pinned to an id. */
  trackedRevision?: string;
  /** The immutable id the fileset is pinned to. */
  revision: string;
  webUrl: string;
}

export const githubCommitUrl = (owner: string, repo: string, revision: string): string =>
  `https://github.com/${owner}/${repo}/commit/${revision}`;

const isGithubStorage = (
  storage: FilesetOutput['storage'] | undefined
): storage is GithubStorageConfig => storage?.type === 'github';

export const agentSpecSource = (
  fileset: FilesetOutput | undefined
): AgentSpecSource | undefined => {
  if (!fileset || !isGithubStorage(fileset.storage)) return undefined;

  // The service pins revision to a resolved commit before it stores the fileset, so it is
  // only optional in the generated type because the model carries a default.
  const { owner, repo, path, revision = 'HEAD', original_revision: tracked } = fileset.storage;
  return {
    owner,
    repo,
    repository: path ? `${owner}/${repo}/${path}` : `${owner}/${repo}`,
    // The service records the requested ref even when it was already a commit, and a ref
    // equal to what it resolved to cannot name anything else — so it is not tracking.
    trackedRevision: tracked && tracked !== revision ? tracked : undefined,
    revision,
    webUrl: `https://github.com/${owner}/${repo}/tree/${revision}${path ? `/${path}` : ''}`,
  };
};

/**
 * The agent's spec fileset. A 404 is a normal answer — an agent registered without
 * one deploys from its inline config — so it resolves to undefined rather than an error.
 */
export const useAgentSpecFileset = (workspace: string, agentName: string | undefined) => {
  const filesetName = agentName ? agentSpecFilesetName(agentName) : '';

  return useFilesRetrieveFileset(workspace, filesetName, {
    query: {
      enabled: Boolean(workspace && filesetName),
      retry: (failureCount, error) => !isNotFoundError(error) && failureCount < 3,
    },
  });
};

/** Re-resolves the fileset's tracked ref, moving it to whatever that ref names now. */
export const useRefreshAgentSpecFileset = (workspace: string, agentName: string | undefined) => {
  const queryClient = useQueryClient();
  const filesetName = agentName ? agentSpecFilesetName(agentName) : '';

  return useMutation({
    mutationFn: () => filesRefreshFileset(workspace, filesetName),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: getFilesRetrieveFilesetQueryKey(workspace, filesetName),
      }),
  });
};
