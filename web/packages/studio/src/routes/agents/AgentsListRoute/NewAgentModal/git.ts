// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { GitStorageConfig } from '@nemo/sdk/generated/platform/schema';
import {
  dropGitSuffix,
  trimSlashes,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/github';

export class GitSourceError extends Error {}

/** Parses the `<ssh-remote>[@<ref>][#<agent_path>]` agent spec. */
export interface GitAgentSource {
  url: string;
  /** Branch, tag, or full commit SHA. Undefined lets the files service resolve the remote's HEAD. */
  ref?: string;
  /** Directory holding agent.yaml. Empty for the repository root. */
  path: string;
}

// Kept in step with parse_ssh_remote in nemo_helix_plugin/files/storage_config.py.
const SSH_USER = String.raw`[A-Za-z0-9._][A-Za-z0-9._-]*`;
const SSH_HOST = String.raw`[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?\.?`;
const SSH_PATH_CHAR = String.raw`[^\s\x00-\x1f\x7f]`;
const SSH_URL = new RegExp(
  String.raw`^ssh://(?:${SSH_USER}@)?(${SSH_HOST})(?::([0-9]{1,5}))?/${SSH_PATH_CHAR}+$`
);
const SCP_REMOTE = new RegExp(
  String.raw`^(?:${SSH_USER}@)?(${SSH_HOST}):[^\s\x00-\x1f\x7f-]${SSH_PATH_CHAR}*$`
);
const MAX_PORT = 65535;

/** Where the remote's path begins, so an `@` past it opens a ref rather than naming a user. */
const pathStart = (locator: string): number => {
  if (locator.startsWith('ssh://')) {
    const slash = locator.indexOf('/', 'ssh://'.length);
    return slash === -1 ? locator.length : slash;
  }
  const colon = locator.indexOf(':');
  return colon === -1 ? locator.length : colon;
};

export const parseGitSource = (input: string): GitAgentSource => {
  const trimmed = input.trim();
  if (!trimmed) throw new GitSourceError('Enter an SSH repository URL.');

  const hash = trimmed.indexOf('#');
  const path = hash === -1 ? '' : trimSlashes(trimmed.slice(hash + 1));
  const locatorAndRef = hash === -1 ? trimmed : trimmed.slice(0, hash);

  const refAt = locatorAndRef.indexOf('@', pathStart(locatorAndRef));
  const url = refAt === -1 ? locatorAndRef : locatorAndRef.slice(0, refAt);
  const ref = refAt === -1 ? undefined : locatorAndRef.slice(refAt + 1) || undefined;

  const port = Number(SSH_URL.exec(url)?.[2] ?? 22);
  if (
    url.includes('%') ||
    !(url.includes('://') ? SSH_URL : SCP_REMOTE).test(url) ||
    port < 1 ||
    port > MAX_PORT
  ) {
    throw new GitSourceError(
      `"${trimmed}" is not an SSH repository. Use git@host:org/repo.git or ssh://host/org/repo.git, optionally with @branch and #sub/directory.`
    );
  }
  return { url, ref, path };
};

/** Human-readable `url[@ref][#path]`, for error text. */
export const formatGitSource = ({ url, ref, path }: GitAgentSource): string =>
  `${url}${ref ? `@${ref}` : ''}${path ? `#${path}` : ''}`;

/** The repository name from the remote's last path segment, without `.git`. */
export const repoNameFromGitUrl = (url: string): string =>
  dropGitSuffix(trimSlashes(url).split(/[/:]/).pop() ?? '');

export const gitStorageConfig = (
  source: GitAgentSource,
  sshKeySecret: string,
  knownHosts: string
): GitStorageConfig => ({
  type: 'git',
  url: source.url,
  ...(source.ref ? { revision: source.ref } : {}),
  ...(source.path ? { path: source.path } : {}),
  ssh_key_secret: sshKeySecret,
  known_hosts: knownHosts.trim(),
});

/** `host[:port]` as ssh names it, with the default port left off. */
export const sshHostOf = (url: string): string | undefined => {
  const match = (url.includes('://') ? SSH_URL : SCP_REMOTE).exec(url);
  // Normalized as the server's allowlist key is: no trailing dot, and the port as a number.
  const host = match?.[1]?.toLowerCase().replace(/\.$/, '');
  if (!host) return undefined;
  const port = match?.[2] ? Number(match[2]) : 22;
  return port === 22 ? host : `${host}:${port}`;
};

const KEY_TYPE_LABELS: Record<string, string> = {
  'ssh-ed25519': 'ED25519',
  'ssh-rsa': 'RSA',
};

/** The algorithm name ssh prints beside a fingerprint, e.g. ED25519 or ECDSA. */
export const sshKeyTypeLabel = (keyType: string): string =>
  KEY_TYPE_LABELS[keyType] ?? (keyType.startsWith('ecdsa-') ? 'ECDSA' : keyType);
