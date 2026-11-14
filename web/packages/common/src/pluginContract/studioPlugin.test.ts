// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CONTRACT_DOCS, diffContract, PLUGIN_CONTRACT } from '@nemo/common/src/pluginContract';
import { studioPlugin } from '@nemo/common/src/pluginContract/studioPlugin';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { build, type Plugin } from 'rolldown';

const INSTALLED: Record<string, string> = {
  react: '19.2.7',
  'react-dom': '19.2.7',
  'react-router': '8.3.0',
  '@tanstack/react-query': '5.90.21',
};

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

const runBuild = (plugin: Plugin = studioPlugin({ root }), dir = root) =>
  build({
    input: path.join(dir, 'src/index.js'),
    cwd: dir,
    platform: 'browser',
    logLevel: 'silent',
    plugins: [plugin],
    output: { dir: path.join(dir, 'dist'), format: 'esm' },
  });

/** Writes `<dir>/node_modules/<pkg>/package.json` (relative to the temp root) for each version. */
const installVersions = (dir: string, versions: Record<string, string>) => {
  for (const [name, version] of Object.entries(versions)) {
    write(`${dir}/node_modules/${name}/package.json`, JSON.stringify({ name, version }));
  }
};

const EXTERNAL_ENTRY = `import a from 'react';\nimport b from 'react-router';\nconsole.log(a, b);\n`;

const builtAgainstAt = (dir: string) =>
  JSON.parse(fs.readFileSync(path.join(root, dir, 'dist/plugin.manifest.json'), 'utf8'))
    .builtAgainst;

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
    const manifest = JSON.parse(fs.readFileSync(manifestPath(), 'utf8'));
    expect(manifest).not.toHaveProperty('bundledPackages');
    expect(manifest).toEqual({
      contractVersion: PLUGIN_CONTRACT.version,
      specifiers: ['@tanstack/react-query', 'react', 'react-router', 'react/jsx-runtime'],
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
    expect(message).toContain(CONTRACT_DOCS);
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

  it('detects a singleton bundled through a pnpm symlink layout', async () => {
    writeFixture(`import x from '../node_modules/react-router/index.js';\nconsole.log(x);\n`);
    const store = 'node_modules/.pnpm/react-router@8.3.0/node_modules/react-router';
    write(`${store}/index.js`, `export default 'react-router';\n`);
    fs.rmSync(path.join(root, 'node_modules/react-router'), { recursive: true });
    fs.symlinkSync(
      path.relative(path.join(root, 'node_modules'), path.join(root, store)),
      path.join(root, 'node_modules/react-router')
    );
    const message = await buildError();

    expect(message).toContain('PLUGIN_SINGLETON_BUNDLED');
    expect(message).toContain('react-router');
    expect(fs.existsSync(manifestPath()), 'no manifest after failed build').toBe(false);
  });

  it('detects a scoped singleton bundled by relative import', async () => {
    writeFixture(
      `import x from '../node_modules/@tanstack/react-query/index.js';\nconsole.log(x);\n`
    );
    const message = await buildError();

    expect(message).toContain('PLUGIN_SINGLETON_BUNDLED');
    expect(message).toContain('@tanstack/react-query');
  });

  it('does not flag packages whose names merely start with a singleton name', async () => {
    writeFixture(
      `import a from '../node_modules/react-dom-foo/index.js';\nimport b from '../node_modules/react-router-dom/index.js';\nconsole.log(a, b);\n`
    );
    installPackage('react-dom-foo', '1.0.0');
    installPackage('react-router-dom', '1.0.0');
    await runBuild();

    const { specifiers } = JSON.parse(fs.readFileSync(manifestPath(), 'utf8'));
    expect(specifiers).not.toContain('react-dom-foo');
    expect(specifiers).not.toContain('react-router-dom');
  });

  it('does not identity-check provided packages bundled by relative import', async () => {
    writeFixture(`import x from '../node_modules/@nemo/common/index.js';\nconsole.log(x);\n`);
    installPackage('@nemo/common', '1.0.0');
    await runBuild();

    expect(fs.existsSync(manifestPath()), 'manifest emitted').toBe(true);
  });

  it('finds hoisted dependencies in ancestor node_modules; the nearest one wins', async () => {
    write('pkg/web/src/index.js', EXTERNAL_ENTRY);
    installVersions('pkg', { react: '19.2.7', 'react-router': '8.3.0' });
    const web = path.join(root, 'pkg/web');
    await runBuild(studioPlugin({ root: web }), web);
    expect(builtAgainstAt('pkg/web'), 'ancestor versions reported').toEqual({
      react: '19.2.7',
      'react-router': '8.3.0',
    });

    installVersions('pkg/web', { react: '18.3.1' });
    await runBuild(studioPlugin({ root: web }), web);
    expect(builtAgainstAt('pkg/web'), 'nested react beats ancestor react').toEqual({
      react: '18.3.1',
      'react-router': '8.3.0',
    });
  });

  it('defaults the root to the Vite config root, but an explicit root wins', async () => {
    for (const dir of ['a', 'b']) {
      write(`${dir}/src/index.js`, EXTERNAL_ENTRY);
    }
    installVersions('a', { react: '19.2.7' });
    installVersions('b', { react: '18.3.1' });
    const [a, b] = [path.join(root, 'a'), path.join(root, 'b')];
    type ViteHooks = { configResolved?: (config: { root: string }) => void };

    const viteRooted = studioPlugin();
    (viteRooted as ViteHooks).configResolved?.({ root: a });
    await runBuild(viteRooted, a);
    expect(builtAgainstAt('a'), 'versions come from config.root').toEqual({ react: '19.2.7' });

    const explicit = studioPlugin({ root: b });
    (explicit as ViteHooks).configResolved?.({ root: a });
    await runBuild(explicit, a);
    expect(builtAgainstAt('a'), 'explicit root beats config.root').toEqual({ react: '18.3.1' });
  });

  it('gives provided-package deep imports the right advice', async () => {
    writeFixture(`import x from '@nemo/common/components/Foo';\nconsole.log(x);\n`);
    const message = await buildError();

    expect(message).toContain('PLUGIN_SPECIFIER_UNMAPPED');
    expect(message, 'advises the bare @nemo/common specifier').toMatch(/[`"']@nemo\/common[`"']/);
    expect(message).toContain('web/packages/common/src/plugin.ts');
    // Wording check by necessity: the advice differs per case, and for the provided package
    // "bundle your own copy" is the wrong fix (it would shadow Studio's @nemo/common).
    expect(message).not.toContain('bundles its own copy');
  });

  it('tells authors to build in production mode when the dev JSX runtime is imported', async () => {
    writeFixture(`import x from 'react/jsx-dev-runtime';\nconsole.log(x);\n`);
    const message = await buildError();

    expect(message).toContain('PLUGIN_SINGLETON_BUNDLED');
    // Wording check by necessity: only this case needs build-mode advice (Studio does not
    // provide the development JSX runtime); the next test guards against it leaking elsewhere.
    expect(message).toContain('production');
    expect(fs.existsSync(manifestPath()), 'no manifest after failed build').toBe(false);
  });

  it('does not mention production mode for other deep imports', async () => {
    writeFixture(`import x from 'react-router/dom';\nconsole.log(x);\n`);
    const message = await buildError();

    expect(message).toContain('PLUGIN_SINGLETON_BUNDLED');
    expect(message).not.toContain('production');
  });

  describe('packages identified by package.json name, not by path', () => {
    const app = () => path.join(root, 'app');
    const buildApp = (entry: string) => {
      write('app/package.json', JSON.stringify({ name: 'my-plugin' }));
      write('app/src/index.js', entry);
      return runBuild(studioPlugin({ root: app() }), app());
    };
    const appBuildError = async (entry: string): Promise<string> => {
      try {
        await buildApp(entry);
      } catch (error) {
        return error instanceof Error ? error.message : String(error);
      }
      return 'BUILD SUCCEEDED';
    };
    const appManifest = () => path.join(app(), 'dist/plugin.manifest.json');
    const importIndex = (rel: string) => `import x from '${rel}';\nconsole.log(x);\n`;

    it('rejects a singleton linked from a checkout outside node_modules', async () => {
      write('lib/react-router/package.json', JSON.stringify({ name: 'react-router' }));
      write('lib/react-router/index.js', `export default 'react-router';\n`);
      fs.mkdirSync(path.join(app(), 'node_modules'), { recursive: true });
      fs.symlinkSync('../../lib/react-router', path.join(app(), 'node_modules/react-router'));
      const message = await appBuildError(importIndex('../node_modules/react-router/index.js'));

      expect(message).toContain('PLUGIN_SINGLETON_BUNDLED');
      expect(message).toContain('react-router');
      expect(fs.existsSync(appManifest()), 'no manifest after failed build').toBe(false);
    });

    it('rejects a plain directory whose package.json names a scoped singleton', async () => {
      write('lib/query/package.json', JSON.stringify({ name: '@tanstack/react-query' }));
      write('lib/query/index.js', `export default 'query';\n`);
      const message = await appBuildError(importIndex('../../lib/query/index.js'));

      expect(message).toContain('PLUGIN_SINGLETON_BUNDLED');
      expect(message).toContain('@tanstack/react-query');
      expect(fs.existsSync(appManifest()), 'no manifest after failed build').toBe(false);
    });

    it.each(['react-router-dom', 'react-dom-foo'])(
      'bundles a directory outside node_modules named like %s',
      async (name) => {
        write(`lib/${name}/package.json`, JSON.stringify({ name }));
        write(`lib/${name}/index.js`, `export default '${name}';\n`);
        await buildApp(importIndex(`../../lib/${name}/index.js`));

        expect(fs.existsSync(appManifest()), 'manifest emitted').toBe(true);
      }
    );

    it("bundles the plugin's own source", async () => {
      write('app/src/local.js', `export default 'local';\n`);
      await buildApp(importIndex('./local.js'));

      expect(fs.existsSync(appManifest()), 'manifest emitted').toBe(true);
    });

    it('bundles a module that has no package.json above it', async () => {
      const orphan = fs.mkdtempSync(path.join(os.tmpdir(), 'studio-plugin-orphan-'));
      try {
        fs.writeFileSync(path.join(orphan, 'index.js'), `export default 'orphan';\n`);
        await buildApp(importIndex(path.join(orphan, 'index.js')));

        expect(fs.existsSync(appManifest()), 'manifest emitted').toBe(true);
      } finally {
        fs.rmSync(orphan, { recursive: true, force: true });
      }
    });
  });
});
