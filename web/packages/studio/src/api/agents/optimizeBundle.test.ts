// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  isFilesetRelative,
  looksLikeOptimizeConfig,
  optimizeBundleProblems,
  parseOptimizeConfig,
} from '@studio/api/agents/optimizeBundle';

const OVERLAY = `
optimizer:
  numeric:
    enabled: true
    n_trials: 2
  search_space:
    temperature:
      type: fabric
      path: models.default.temperature
      values: [0.0, 0.2]
eval:
  general:
    dataset:
      file_path: dataset-chatonly.json
  fabric:
    base_dir: .
`;

const configOf = (text: string): Record<string, unknown> => {
  const config = parseOptimizeConfig(text);
  if (!config) throw new Error('fixture is not a YAML mapping');
  return config;
};

const DEFAULT_PATHS = ['optimize.yaml', 'dataset-chatonly.json'];

const problemsFor = (text: string, paths: string[] = DEFAULT_PATHS) =>
  optimizeBundleProblems({ config: configOf(text), bundlePaths: new Set(paths), agent: 'hermes' });

const problemsWithoutAgent = (text: string) =>
  optimizeBundleProblems({ config: configOf(text), bundlePaths: new Set(DEFAULT_PATHS) });

describe('optimizeBundleProblems', () => {
  it('passes the hermes chat-only overlay when an agent is given', () => {
    expect(problemsFor(OVERLAY)).toEqual([]);
  });

  it('requires an Agent under Test when no agent is given', () => {
    expect(problemsWithoutAgent(OVERLAY)).toEqual([expect.stringContaining('no Agent under Test')]);
  });

  it('accepts an inline Fabric package without an agent', () => {
    expect(problemsWithoutAgent(`schema_version: fabric.agent/v1alpha1\n${OVERLAY}`)).toEqual([]);
  });

  it('names legacy NAT workflow YAML', () => {
    expect(problemsWithoutAgent(`workflow: {}\n${OVERLAY}`)).toEqual([
      expect.stringContaining('legacy NAT workflow YAML'),
    ]);
  });

  it('requires an enabled optimizer with a search space', () => {
    expect(problemsFor('optimizer:\n  numeric:\n    enabled: false\n')).toEqual([
      expect.stringContaining('no optimizer is enabled'),
    ]);
    expect(problemsFor('optimizer:\n  numeric:\n    enabled: true\n  search_space: {}\n')).toEqual([
      expect.stringContaining('optimizer.search_space is empty'),
    ]);
  });

  it('reports a dataset missing from the bundle', () => {
    expect(problemsFor(OVERLAY, ['optimize.yaml'])).toEqual([
      expect.stringContaining('eval.general.dataset points at "dataset-chatonly.json"'),
    ]);
  });

  it('reports absolute paths even when the file is in the bundle', () => {
    const text = OVERLAY.replace('file_path: dataset-chatonly.json', 'file_path: /tmp/data.json');
    expect(problemsFor(text)).toEqual([expect.stringContaining('is an absolute path')]);
  });

  it('skips fileset-ref datasets and templated paths', () => {
    expect(
      problemsFor(OVERLAY.replace('dataset-chatonly.json', 'default/data#rows.json'), [
        'optimize.yaml',
      ])
    ).toEqual([]);
    expect(
      problemsFor(OVERLAY.replace('dataset-chatonly.json', '${DATA_DIR}/rows.json'), [
        'optimize.yaml',
      ])
    ).toEqual([]);
  });

  it('checks base_dir as a directory', () => {
    const text = OVERLAY.replace('base_dir: .', 'base_dir: ./agents/chatonly');
    expect(problemsFor(text)).toEqual([expect.stringContaining('not a directory in the bundle')]);
    expect(
      problemsFor(text, ['optimize.yaml', 'dataset-chatonly.json', 'agents/chatonly/agent.yaml'])
    ).toEqual([]);
  });

  it('checks stdio MCP scripts but not commands or flags', () => {
    const text = `${OVERLAY}
mcp:
  servers:
    analyzer:
      transport: stdio
      url: python3
      args: ["-u", "phishing_analyzer_mcp/server.py"]
`;
    expect(problemsFor(text)).toEqual([
      expect.stringContaining('mcp.servers.analyzer.args[1] points at'),
    ]);
    expect(
      problemsFor(text, [
        'optimize.yaml',
        'dataset-chatonly.json',
        'phishing_analyzer_mcp/server.py',
      ])
    ).toEqual([]);
  });

  it('only requires output directories to be relative', () => {
    expect(problemsFor(`${OVERLAY}runtime:\n  artifacts: out/\n`)).toEqual([]);
    expect(problemsFor(`${OVERLAY}runtime:\n  artifacts: /var/out\n`)).toEqual([
      expect.stringContaining('runtime.artifacts is an absolute path'),
    ]);
  });
});

describe('isFilesetRelative', () => {
  it.each(['a.yaml', 'dir/a.yaml', './a.yaml'])('accepts %s', (path) => {
    expect(isFilesetRelative(path)).toBe(true);
  });

  it.each(['/a.yaml', '~/a.yaml', 'C:\\a.yaml', 'D:a.yaml', '../a.yaml', 'dir\\..\\a.yaml'])(
    'rejects %s',
    (path) => {
      expect(isFilesetRelative(path)).toBe(false);
    }
  );
});

describe('looksLikeOptimizeConfig', () => {
  it('needs an optimizer mapping', () => {
    expect(looksLikeOptimizeConfig(parseOptimizeConfig(OVERLAY))).toBe(true);
    expect(looksLikeOptimizeConfig(parseOptimizeConfig('name: agent\n'))).toBe(false);
    expect(looksLikeOptimizeConfig(parseOptimizeConfig('- a\n- b\n'))).toBe(false);
    expect(looksLikeOptimizeConfig(parseOptimizeConfig('key: [unclosed'))).toBe(false);
  });
});
