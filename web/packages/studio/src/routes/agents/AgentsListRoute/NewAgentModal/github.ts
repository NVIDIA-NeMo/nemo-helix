// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { GithubStorageConfig } from '@nemo/sdk/generated/platform/schema';

const GITHUB_HOSTS = new Set(['github.com', 'www.github.com']);

export class GitHubSourceError extends Error {}

/** Mirrors the Experimentalist plugin's `<git-url>[@<ref>][#<agent_path>]` agent spec. */
export interface GitHubAgentSource {
  owner: string;
  repo: string;
  /** Branch, tag, or commit. Undefined lets the files service resolve the default branch. */
  ref?: string;
  /** Directory holding agent.yaml. Empty for the repository root. */
  path: string;
}

const trimSlashes = (value: string): string => value.replace(/^\/+|\/+$/g, '');

const dropGitSuffix = (value: string): string => value.replace(/\.git$/, '');

/** Repository path segments, or undefined when this is not a GitHub locator. */
const githubPathSegments = (locator: string): string[] | undefined => {
  if (locator.includes('://')) {
    let url: URL;
    try {
      url = new URL(locator);
    } catch {
      return undefined;
    }
    return GITHUB_HOSTS.has(url.hostname.toLowerCase())
      ? trimSlashes(url.pathname).split('/')
      : undefined;
  }

  // SCP form, `[user@]github.com:owner/repo`.
  const colon = locator.indexOf(':');
  if (colon !== -1) {
    const host = locator.slice(0, colon).split('@').pop() ?? '';
    if (!GITHUB_HOSTS.has(host.toLowerCase())) return undefined;
    return trimSlashes(locator.slice(colon + 1)).split('/');
  }

  const bare = trimSlashes(locator);
  return GITHUB_HOSTS.has(bare.split('/')[0]?.toLowerCase() ?? '')
    ? bare.split('/').slice(1)
    : bare.split('/');
};

/**
 * Accepts `owner/repo`, an HTTPS or SSH clone URL, and a `/tree/<ref>/<path>` browser URL,
 * each optionally suffixed `@<ref>` and `#<path>`.
 *
 * A branch containing a slash is indistinguishable from a nested path in a `/tree/` URL, so
 * the first segment after `tree` is taken as the ref. Use `@<ref>` to be explicit.
 */
export const parseGitHubSource = (input: string): GitHubAgentSource => {
  const trimmed = input.trim();
  if (!trimmed) throw new GitHubSourceError('Enter a GitHub repository URL.');

  const hash = trimmed.indexOf('#');
  const fragmentPath = hash === -1 ? '' : trimSlashes(trimmed.slice(hash + 1));
  const locatorAndRef = hash === -1 ? trimmed : trimmed.slice(0, hash);

  // Only an `@` in the last segment marks a ref; earlier ones are the `git@host` userinfo.
  const lastSlash = locatorAndRef.lastIndexOf('/');
  const refAt = locatorAndRef.indexOf('@', lastSlash + 1);
  const explicitRef = refAt === -1 ? undefined : locatorAndRef.slice(refAt + 1) || undefined;
  const locator = refAt === -1 ? locatorAndRef : locatorAndRef.slice(0, refAt);

  const segments = githubPathSegments(locator)?.filter(Boolean);
  if (!segments || segments.length < 2) {
    throw new GitHubSourceError(
      `"${trimmed}" is not a GitHub repository. Use github.com/owner/repo, optionally with @branch and #sub/directory.`
    );
  }

  const [owner, rawRepo, kind, ...rest] = segments;
  const repo = dropGitSuffix(rawRepo ?? '');
  if (!owner || !repo) {
    throw new GitHubSourceError(`"${trimmed}" is missing an owner or a repository name.`);
  }

  // Browser URLs: /tree/<ref>/<path> and /blob/<ref>/<path>.
  const browsed = kind === 'tree' || kind === 'blob';
  const urlRef = browsed ? rest[0] : undefined;
  const urlPath = browsed ? rest.slice(1).join('/') : segments.slice(2).join('/');

  return {
    owner,
    repo,
    ref: explicitRef ?? urlRef,
    path: fragmentPath || trimSlashes(urlPath),
  };
};

/** Human-readable `owner/repo[@ref][#path]`, for error text and the selection summary. */
export const formatGitHubSource = ({ owner, repo, ref, path }: GitHubAgentSource): string =>
  `${owner}/${repo}${ref ? `@${ref}` : ''}${path ? `#${path}` : ''}`;

/**
 * The spec fileset reads the repository directly, so the token stays in the files service
 * and is never fetched into the browser.
 */
export const githubStorageConfig = (
  source: GitHubAgentSource,
  secretName?: string
): GithubStorageConfig => ({
  type: 'github',
  owner: source.owner,
  repo: source.repo,
  ...(source.ref ? { revision: source.ref } : {}),
  ...(source.path ? { path: source.path } : {}),
  ...(secretName ? { token_secret: secretName } : {}),
});

/** A starting point for the agent name, which the user can still edit before submitting. */
export const agentNameFromSource = (source: GitHubAgentSource): string => {
  const candidate = source.path ? (source.path.split('/').pop() ?? source.repo) : source.repo;
  return candidate
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, '-')
    .replace(/^-+|-+$/g, '');
};
