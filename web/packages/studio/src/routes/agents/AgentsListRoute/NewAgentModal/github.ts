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

/** Browser URLs that carry `/<ref>/<path>` after the repository. */
const REF_BEARING_ROUTES = new Set(['tree', 'blob', 'raw', 'blame', 'commit']);

/** Browser URLs that name something other than repository content. */
const NON_CONTENT_ROUTES = new Set([
  'actions',
  'branches',
  'commits',
  'compare',
  'discussions',
  'issues',
  'pull',
  'pulls',
  'releases',
  'security',
  'settings',
  'tags',
  'wiki',
]);

const SURROUNDING_SLASHES = /^\/+|\/+$/g;
const GIT_SUFFIX = /\.git$/;

const trimSlashes = (value: string): string => value.replace(SURROUNDING_SLASHES, '');

const dropGitSuffix = (value: string): string => value.replace(GIT_SUFFIX, '');

/**
 * Where the host ends. An `@` before this is `user@host` userinfo; one after it opens a ref,
 * which may itself contain slashes.
 */
const authorityEnd = (locator: string): number => {
  const scheme = locator.indexOf('://');
  const start = scheme === -1 ? 0 : scheme + 3;
  const slash = locator.indexOf('/', start);
  // Only the SCP form separates the host with a colon; in a URL a colon is the port.
  const colon = scheme === -1 ? locator.indexOf(':', start) : -1;
  const ends = [slash, colon].filter((index) => index !== -1);
  return ends.length > 0 ? Math.min(...ends) : locator.length;
};

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

  const segments = trimSlashes(locator).split('/');
  const first = segments[0]?.toLowerCase() ?? '';
  if (GITHUB_HOSTS.has(first)) return segments.slice(1);
  // A GitHub owner is alphanumerics and hyphens, so a dot here is a host — and not ours.
  return first.includes('.') ? undefined : segments;
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

  const refAt = locatorAndRef.indexOf('@', authorityEnd(locatorAndRef));
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

  if (kind && NON_CONTENT_ROUTES.has(kind)) {
    throw new GitHubSourceError(
      `"${trimmed}" points at ${owner}/${repo}'s ${kind}, not at its files. Use the repository URL, or a /tree/ URL for a branch and directory.`
    );
  }

  const browsed = Boolean(kind && REF_BEARING_ROUTES.has(kind));
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
