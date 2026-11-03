// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import {
  filesRefreshFileset,
  getFilesRetrieveFilesetQueryKey,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import type {
  FilesetOutput,
  GitStorageConfig,
  GithubStorageConfig,
} from '@nemo/sdk/generated/platform/schema';
import { agentSpecFilesetName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import {
  type UseMutationResult,
  type UseQueryResult,
  useMutation,
  useQueryClient,
} from '@tanstack/react-query';

export interface AgentSpecSource {
  /** `owner/repo` or the SSH remote, with the sub-directory appended when the fileset is scoped to one. */
  repository: string;
  /** The mutable ref the fileset tracks, if any. Absent when it was pinned to an id. */
  trackedRevision?: string;
  /** The immutable id the fileset is pinned to. */
  revision: string;
  /** The pinned tree on the web. Absent for a repository read over SSH, which has no web view to point at. */
  webUrl?: string;
  /** A commit on the web. Absent for a repository read over SSH. */
  commitUrl?: (revision: string) => string;
}

/** Encode a repository path, keeping the separators between its segments. */
const encodePath = (value: string): string => value.split('/').map(encodeURIComponent).join('/');

export const githubCommitUrl = (owner: string, repo: string, revision: string): string =>
  `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/commit/${encodeURIComponent(revision)}`;

const isGithubStorage = (
  storage: FilesetOutput['storage'] | undefined
): storage is GithubStorageConfig => storage?.type === 'github';

const isGitStorage = (storage: FilesetOutput['storage'] | undefined): storage is GitStorageConfig =>
  storage?.type === 'git';

// The service records the requested ref even when it was already a commit, and a ref
// equal to what it resolved to cannot name anything else — so it is not tracking.
const trackedRevisionOf = (tracked: string | undefined, revision: string): string | undefined =>
  tracked && tracked !== revision ? tracked : undefined;

export const agentSpecSource = (
  fileset: FilesetOutput | undefined
): AgentSpecSource | undefined => {
  if (!fileset) return undefined;

  // The service pins revision to a resolved commit before it stores the fileset, so it is
  // only optional in the generated type because the model carries a default.
  if (isGithubStorage(fileset.storage)) {
    const { owner, repo, path, revision = 'HEAD', original_revision: tracked } = fileset.storage;
    return {
      repository: path ? `${owner}/${repo}/${path}` : `${owner}/${repo}`,
      trackedRevision: trackedRevisionOf(tracked, revision),
      revision,
      webUrl: `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/tree/${encodeURIComponent(revision)}${path ? `/${encodePath(path)}` : ''}`,
      commitUrl: (commit) => githubCommitUrl(owner, repo, commit),
    };
  }
  if (isGitStorage(fileset.storage)) {
    const { url, path, revision = 'HEAD', original_revision: tracked } = fileset.storage;
    return {
      repository: path ? `${url}#${path}` : url,
      trackedRevision: trackedRevisionOf(tracked, revision),
      revision,
    };
  }
  return undefined;
};

/**
 * The agent's spec fileset. A 404 is a normal answer — an agent registered without
 * one deploys from its inline config — so it resolves to undefined rather than an error.
 */
export const useAgentSpecFileset = (
  workspace: string,
  agentName: string | undefined
): UseQueryResult<FilesetOutput | undefined, Error | null> => {
  const filesetName = agentName ? agentSpecFilesetName(agentName) : '';

  return useFilesRetrieveFileset(workspace, filesetName, {
    query: {
      enabled: Boolean(workspace && filesetName),
      retry: (failureCount, error) => !isNotFoundError(error) && failureCount < 3,
    },
  });
};

/** Re-resolves the fileset's tracked ref, moving it to whatever that ref names now. */
export const useRefreshAgentSpecFileset = (
  workspace: string,
  agentName: string | undefined
): UseMutationResult<FilesetOutput, Error, void> => {
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
