// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { REPOSITORY_EXAMPLES } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import {
  GitSourceError,
  gitStorageConfig,
  parseGitSource,
  repoNameFromGitUrl,
  sshHostOf,
  sshKeyTypeLabel,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/git';
import {
  agentNameFromRepository,
  isSshRemote,
  parseRepositorySource,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/repository';

describe('parseGitSource', () => {
  it.each([
    [
      'git@gitlab.example.com:acme/agents.git',
      'git@gitlab.example.com:acme/agents.git',
      undefined,
      '',
    ],
    [
      'git@host:acme/agents.git@release/1.0#calc/',
      'git@host:acme/agents.git',
      'release/1.0',
      'calc',
    ],
    ['ssh://git@host:2222/acme/agents.git@main', 'ssh://git@host:2222/acme/agents.git', 'main', ''],
    ['host:acme/agents#a/b', 'host:acme/agents', undefined, 'a/b'],
  ])('reads %s', (input, url, ref, path) => {
    expect(parseGitSource(input)).toEqual({ url, ref, path });
  });

  it.each([
    '-oProxyCommand=x:repo',
    'git@-host:repo',
    'ssh://host',
    'not a url',
    'ssh://allowed.example?.evil.example/acme/agents.git',
    'allowed.example#.evil.example:acme/agents.git',
    'ssh://evil.example%2F@allowed.example/acme/agents.git',
    'ssh://host.example:70000/acme/agents.git',
    'ssh://host.example:0/acme/agents.git',
    'ssh://host.example:\u0662\u0662/acme/agents.git',
  ])('rejects %s', (input) => {
    expect(() => parseGitSource(input)).toThrow(GitSourceError);
  });
});

it('accepts an absolute SCP path, as git does', () => {
  expect(parseGitSource('git@host.example:/srv/git/agents.git').url).toBe(
    'git@host.example:/srv/git/agents.git'
  );
});

describe('gitStorageConfig', () => {
  it('omits the ref and path it was not given and trims the host keys', () => {
    expect(
      gitStorageConfig({ url: 'git@host:a/b.git', path: '' }, 'key', ' host ssh-ed25519 AAAA\n')
    ).toEqual({
      type: 'git',
      url: 'git@host:a/b.git',
      ssh_key_secret: 'key',
      known_hosts: 'host ssh-ed25519 AAAA',
    });
  });
});

describe('repoNameFromGitUrl', () => {
  it.each([
    ['git@host:acme/agents.git', 'agents'],
    ['host:agents', 'agents'],
    ['ssh://host:22/acme/calc-bot/', 'calc-bot'],
  ])('names %s', (url, name) => {
    expect(repoNameFromGitUrl(url)).toBe(name);
  });
});

describe('parseRepositorySource', () => {
  it.each([
    ['git@github.com:acme/agents.git', true],
    ['ssh://host/acme/agents', true],
    ['host:acme/agents', true],
    ['github.com/acme/agents', false],
    ['acme/agents', false],
    ['https://github.com/acme/agents', false],
  ])('treats %s as SSH: %s', (input, ssh) => {
    expect(isSshRemote(input)).toBe(ssh);
  });

  it('reads SSH remotes with the SSH parser and everything else as GitHub', () => {
    expect(parseRepositorySource('git@host:acme/agents.git').kind).toBe('ssh');
    expect(parseRepositorySource('github.com/acme/agents')).toEqual({
      kind: 'github',
      source: { owner: 'acme', repo: 'agents', ref: undefined, path: '' },
    });
  });

  it('points a non-GitHub HTTPS URL at the SSH form', () => {
    expect(() => parseRepositorySource('https://gitlab.com/acme/agents')).toThrow(
      /is not a GitHub repository.*use an SSH URL/
    );
  });

  it('names the agent after the directory, then the repository', () => {
    expect(agentNameFromRepository(parseRepositorySource('git@host:acme/Calc_Bot.git'))).toBe(
      'calc-bot'
    );
    expect(
      agentNameFromRepository(parseRepositorySource('git@host:acme/agents.git#x/support'))
    ).toBe('support');
  });
});

describe('sshHostOf', () => {
  it.each([
    ['git@GitLab.example.com:acme/agents.git', 'gitlab.example.com'],
    ['ssh://git@gitlab-master.nvidia.com:12051/acme/agents.git', 'gitlab-master.nvidia.com:12051'],
    ['ssh://host:22/acme/agents', 'host'],
    ['ssh://host:022/acme/agents', 'host'],
    ['ssh://host.example.:2222/acme/agents', 'host.example:2222'],
  ])('reads %s', (url, host) => {
    expect(sshHostOf(url)).toBe(host);
  });
});

describe('sshKeyTypeLabel', () => {
  it.each([
    ['ssh-ed25519', 'ED25519'],
    ['ecdsa-sha2-nistp256', 'ECDSA'],
    ['ssh-rsa', 'RSA'],
  ])('labels %s as %s', (keyType, label) => {
    expect(sshKeyTypeLabel(keyType)).toBe(label);
  });
});

describe('REPOSITORY_EXAMPLES', () => {
  it.each(REPOSITORY_EXAMPLES.map(({ label, value }) => [label, value]))(
    'accepts the %s example',
    (_label, value) => {
      expect(() => parseRepositorySource(value)).not.toThrow();
    }
  );

  it('reads the branch and directory out of the examples that carry them', () => {
    expect(
      parseRepositorySource('ssh://git@gitlab.example.com:2222/acme/agents.git@release/1.0')
    ).toEqual({
      kind: 'ssh',
      source: {
        url: 'ssh://git@gitlab.example.com:2222/acme/agents.git',
        ref: 'release/1.0',
        path: '',
      },
    });
  });
});
