// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { diffContract, PLUGIN_CONTRACT } from '@nemo/common/src/pluginContract';
import { studioPlugin } from '@nemo/common/src/pluginContract/studioPlugin';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { build } from 'rolldown';

const INSTALLED: Record<string, string> = {
  react: '19.2.7',
  'react-dom': '19.2.7',
  'react-router': '8.3.0',
  '@tanstack/react-query': '5.90.21',
};

const DOCS = 'plugins/example-plugin/web/AGENTS.md';

let root: string;

const write = (rel: string, content: string) => {
  const file = path.join(root, rel);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
};

const installPackage = (name: string, version: string, extra: object = {}) => {
  write(
    `node_modules/${name}/package.json`,
    JSON.stringify({ name, version, type: 'module', main: 'index.js', ...extra })
  );
  write(`node_modules/${name}/index.js`, `export default '${name}';\n`);
};

const writeFixture = (entry: string) => {
  write('package.json', JSON.stringify({ name: 'fixture', dependencies: { react: '^19.0.0' } }));
  write('src/index.js', entry);
  write('src/local.js', `export const local = 'local';\n`);
  for (const [name, version] of Object.entries(INSTALLED)) {
    installPackage(name, version);
  }
  write('node_modules/react/jsx-runtime.js', `export const jsx = 1;\n`);
  installPackage('react-router', INSTALLED['react-router'], {
    exports: { '.': './index.js', './dom': './dom.js' },
  });
  write('node_modules/react-router/dom.js', `export default 'dom';\n`);
  installPackage('tiny-util', '1.0.0');
};

const outDir = () => path.join(root, 'dist');

const runBuild = () =>
  build({
    input: path.join(root, 'src/index.js'),
    cwd: root,
    platform: 'browser',
    logLevel: 'silent',
    plugins: [studioPlugin({ root })],
    output: { dir: outDir(), format: 'esm' },
  });

const buildError = async (): Promise<string> => {
  try {
    await runBuild();
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
  return 'BUILD SUCCEEDED';
};

const manifestPath = () => path.join(outDir(), 'plugin.manifest.json');

const CLEAN_ENTRY = `
import React from 'react';
import { jsx } from 'react/jsx-runtime';
import router from 'react-router';
import query from '@tanstack/react-query';
import tiny from 'tiny-util';
import { local } from './local.js';
console.log(React, jsx, router, query, tiny, local);
`;

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'studio-plugin-'));
});

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true });
});

describe('studioPlugin', () => {
  it('externalizes contract specifiers, bundles the rest, and writes the manifest', async () => {
    writeFixture(CLEAN_ENTRY);
    await runBuild();

    const output = fs.readFileSync(path.join(outDir(), 'index.js'), 'utf8');
    for (const specifier of [
      'react',
      'react/jsx-runtime',
      'react-router',
      '@tanstack/react-query',
    ]) {
      expect(output, `bare import of ${specifier} kept`).toMatch(
        new RegExp(`from ['"]${specifier}['"]`)
      );
    }
    expect(output, 'tiny-util is bundled, not left as an import').not.toMatch(
      /from ['"]tiny-util['"]/
    );

    expect(fs.existsSync(manifestPath()), 'plugin.manifest.json emitted').toBe(true);
    expect(JSON.parse(fs.readFileSync(manifestPath(), 'utf8'))).toEqual({
      contractVersion: PLUGIN_CONTRACT.version,
      specifiers: ['@tanstack/react-query', 'react', 'react-router', 'react/jsx-runtime'],
      bundledPackages: [],
      builtAgainst: INSTALLED,
    });
  });

  it.each([
    ['react-router/dom', 'PLUGIN_SINGLETON_BUNDLED', 'react-router'],
    ['@nemo/common/components/Foo', 'PLUGIN_SPECIFIER_UNMAPPED', '@nemo/common/components/Foo'],
  ])('fails the build on deep import %s', async (specifier, code, subject) => {
    writeFixture(`import x from '${specifier}';\nconsole.log(x);\n`);
    const message = await buildError();

    expect(message).toContain(code);
    expect(message).toContain(subject);
    expect(message).toMatch(/1\./);
    expect(message).toContain(DOCS);
    expect(fs.existsSync(manifestPath()), 'no manifest after failed build').toBe(false);
  });

  it('fails after bundling when a singleton source is reached by a relative import', async () => {
    writeFixture(`import x from '../node_modules/react-router/index.js';\nconsole.log(x);\n`);
    const message = await buildError();

    expect(message).toContain('PLUGIN_SINGLETON_BUNDLED');
    expect(message).toContain('react-router');
    expect(fs.existsSync(manifestPath()), 'no manifest after failed build').toBe(false);
  });

  it('emits a manifest that diffContract accepts, and rejects a newer contract version', async () => {
    writeFixture(CLEAN_ENTRY);
    await runBuild();
    const manifest = JSON.parse(fs.readFileSync(manifestPath(), 'utf8'));

    expect(diffContract(manifest, PLUGIN_CONTRACT, INSTALLED)).toEqual([]);
    expect(
      diffContract({ ...manifest, contractVersion: '2.0.0' }, PLUGIN_CONTRACT, INSTALLED).map(
        (i) => i.code
      )
    ).toEqual(['PLUGIN_CONTRACT_UNSUPPORTED']);
  });
});
