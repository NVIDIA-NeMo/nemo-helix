// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  GitHubSourceError,
  agentNameFromSource,
  formatGitHubSource,
  githubStorageConfig,
  parseGitHubSource,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/github';

describe('parseGitHubSource', () => {
  it.each([
    ['owner/repo', { owner: 'owner', repo: 'repo', ref: undefined, path: '' }],
    ['github.com/owner/repo', { owner: 'owner', repo: 'repo', ref: undefined, path: '' }],
    ['https://github.com/owner/repo', { owner: 'owner', repo: 'repo', ref: undefined, path: '' }],
    [
      'https://github.com/owner/repo.git',
      { owner: 'owner', repo: 'repo', ref: undefined, path: '' },
    ],
    ['git@github.com:owner/repo.git', { owner: 'owner', repo: 'repo', ref: undefined, path: '' }],
    [
      'https://github.com/owner/repo/tree/main/agents/calc',
      { owner: 'owner', repo: 'repo', ref: 'main', path: 'agents/calc' },
    ],
    [
      'github.com/owner/repo@v1.2#agents/calc',
      { owner: 'owner', repo: 'repo', ref: 'v1.2', path: 'agents/calc' },
    ],
  ])('parses %s', (input, expected) => {
    expect(parseGitHubSource(input)).toEqual(expected);
  });

  it('does not read the git@ userinfo as a ref', () => {
    expect(parseGitHubSource('git@github.com:owner/repo').ref).toBeUndefined();
  });

  it('rejects a host that is not GitHub', () => {
    expect(() => parseGitHubSource('https://gitlab.com/owner/repo')).toThrow(GitHubSourceError);
  });

  it('rejects input that names no repository', () => {
    expect(() => parseGitHubSource('owner')).toThrow(GitHubSourceError);
    expect(() => parseGitHubSource('   ')).toThrow(GitHubSourceError);
  });
});

describe('formatGitHubSource', () => {
  it('round-trips the spec form', () => {
    expect(formatGitHubSource({ owner: 'o', repo: 'r', ref: 'main', path: 'a/b' })).toBe(
      'o/r@main#a/b'
    );
    expect(formatGitHubSource({ owner: 'o', repo: 'r', path: '' })).toBe('o/r');
  });
});

describe('githubStorageConfig', () => {
  it('omits the revision and path when the source names neither', () => {
    expect(githubStorageConfig({ owner: 'o', repo: 'r', path: '' })).toEqual({
      type: 'github',
      owner: 'o',
      repo: 'r',
    });
  });

  it('carries the revision, directory, and token secret', () => {
    expect(
      githubStorageConfig({ owner: 'o', repo: 'r', ref: 'v1', path: 'agents/calc' }, 'github-pat')
    ).toEqual({
      type: 'github',
      owner: 'o',
      repo: 'r',
      revision: 'v1',
      path: 'agents/calc',
      token_secret: 'github-pat',
    });
  });

  it('omits the token secret for a public repository', () => {
    expect(githubStorageConfig({ owner: 'o', repo: 'r', path: '' })).not.toHaveProperty(
      'token_secret'
    );
  });
});

describe('agentNameFromSource', () => {
  it('uses the repository when no directory is given', () => {
    expect(agentNameFromSource({ owner: 'acme', repo: 'Calc-Agent', path: '' })).toBe('calc-agent');
  });

  it('prefers the directory holding agent.yaml', () => {
    expect(agentNameFromSource({ owner: 'acme', repo: 'agents', path: 'agents/Calc_Bot' })).toBe(
      'calc-bot'
    );
  });

  it('produces a name the form schema accepts', () => {
    expect(agentNameFromSource({ owner: 'acme', repo: '__weird__', path: '' })).toBe('weird');
  });
});
