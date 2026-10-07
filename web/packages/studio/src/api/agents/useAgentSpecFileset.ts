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
import { AGENT_SPEC_DIR_FIELD } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import { agentSpecFilesetName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import {
  type UseMutationResult,
  type UseQueryResult,
  useMutation,
  useQueryClient,
} from '@tanstack/react-query';

export interface AgentSpecSource {
  /** Set for a GitHub fileset, whose commits and tree can be linked. */
  github?: { owner: string; repo: string };
  /** `owner/repo` or the SSH remote, with the sub-directory appended when the fileset is scoped to one. */
  repository: string;
  /** The mutable ref the fileset tracks, if any. Absent when it was pinned to an id. */
  trackedRevision?: string;
  /** The immutable id the fileset is pinned to. */
  revision: string;
  webUrl?: string;
}

/** Encode a repository path, keeping the separators between its segments. */
const encodePath = (value: string): string => value.split('/').map(encodeURIComponent).join('/');

export const githubCommitUrl = (owner: string, repo: string, revision: string): string =>
  `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/commit/${encodeURIComponent(revision)}`;

const isRepositoryStorage = (
  storage: FilesetOutput['storage'] | undefined
): storage is GithubStorageConfig | GitStorageConfig =>
  storage?.type === 'github' || storage?.type === 'git';

export const agentSpecSource = (
  fileset: FilesetOutput | undefined
): AgentSpecSource | undefined => {
  if (!fileset || !isRepositoryStorage(fileset.storage)) return undefined;

  const storage = fileset.storage;
  // The service pins revision to a resolved commit before it stores the fileset, so it is
  // only optional in the generated type because the model carries a default.
  const { path, revision = 'HEAD', original_revision: tracked } = storage;
  // A ref equal to the commit it resolved to cannot move, so it is not tracking anything.
  const trackedRevision = tracked && tracked !== revision ? tracked : undefined;

  // `type` is optional in the generated types, so it does not narrow the union; `url` does.
  if ('url' in storage) {
    const specDir = fileset.custom_fields?.[AGENT_SPEC_DIR_FIELD];
    const directory = typeof specDir === 'string' && specDir ? specDir : path;
    return {
      repository: directory ? `${storage.url}#${directory}` : storage.url,
      trackedRevision,
      revision,
    };
  }

  const { owner, repo } = storage;
  return {
    github: { owner, repo },
    repository: path ? `${owner}/${repo}/${path}` : `${owner}/${repo}`,
    trackedRevision,
    revision,
    webUrl: `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/tree/${encodeURIComponent(revision)}${path ? `/${encodePath(path)}` : ''}`,
  };
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
