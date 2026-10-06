// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type GitAgentSource,
  formatGitSource,
  parseGitSource,
  repoNameFromGitUrl,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/git';
import {
  type GitHubAgentSource,
  agentNameFromSource,
  formatGitHubSource,
  parseGitHubSource,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/github';

export type RepositorySource =
  | { kind: 'ssh'; source: GitAgentSource }
  | { kind: 'github'; source: GitHubAgentSource };

/** `ssh://…`, or SCP form `[user@]host:path` — a colon before any slash and no scheme. */
export const isSshRemote = (input: string): boolean => {
  const trimmed = input.trim();
  if (trimmed.startsWith('ssh://')) return true;
  if (trimmed.includes('://')) return false;
  const colon = trimmed.indexOf(':');
  const slash = trimmed.indexOf('/');
  return colon > 0 && (slash === -1 || colon < slash);
};

/** SSH remotes are read with a private key; anything else goes to the GitHub API, the only HTTPS host. */
export const parseRepositorySource = (input: string): RepositorySource => {
  if (isSshRemote(input)) return { kind: 'ssh', source: parseGitSource(input) };
  try {
    return { kind: 'github', source: parseGitHubSource(input) };
  } catch (error) {
    throw new Error(
      `${(error as Error).message} For other hosts, use an SSH URL like git@host:org/repo.git.`,
      { cause: error }
    );
  }
};

export const agentNameFromRepository = ({ kind, source }: RepositorySource): string =>
  agentNameFromSource(
    kind === 'ssh' ? { repo: repoNameFromGitUrl(source.url), path: source.path } : source
  );

/** The same repository and ref, scoped to a different directory. */
export const withRepositoryPath = (repository: RepositorySource, path: string): RepositorySource =>
  repository.kind === 'ssh'
    ? { kind: 'ssh', source: { ...repository.source, path } }
    : { kind: 'github', source: { ...repository.source, path } };

/** Back to the text the Repository field accepts. */
export const formatRepositorySource = (repository: RepositorySource): string =>
  repository.kind === 'ssh'
    ? formatGitSource(repository.source)
    : formatGitHubSource(repository.source);
