// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { baseTestConfig } from '@nemo/testing/react/config';
import tailwindPostcss from '@tailwindcss/postcss';
import react from '@vitejs/plugin-react';
import * as fs from 'node:fs';
import { createRequire } from 'node:module';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build as rolldownBuild, type Plugin as RolldownPlugin } from 'rolldown';
import license, { type Dependency, type Person } from 'rollup-plugin-license';
import { visualizer } from 'rollup-plugin-visualizer';
import { loadEnv, type Plugin } from 'vite';
import mkcert from 'vite-plugin-mkcert';
import svgr from 'vite-plugin-svgr';
// vite does not know about vitest -- vitest config extends vite config
import { defineConfig, mergeConfig } from 'vitest/config';

interface PackageJson {
  readonly name?: string;
  readonly version?: string;
  readonly license?: string | { readonly type?: string };
  readonly author?: string | { readonly name?: string };
  readonly dependencies?: Record<string, string>;
}

interface LicenseReportDependency {
  readonly name: string | null;
  readonly version: string | null;
  readonly license: string | null;
  readonly author: string | Person | null;
  readonly licenseText: string | null;
}

// Shared deps externalized from Studio's and plugins' production builds. At
// runtime, bare imports of these names resolve via an import map in index.html
// to bundles in public/vendor/, so Studio and every loaded plugin use one
// shared React/react-dom/router instance.
const VENDOR_EXTERNALS = [
  'react',
  'react/jsx-runtime',
  'react-dom',
  'react-dom/client',
  'react-router',
  // The design system is shared so plugins render KUI components against the
  // same theme context Studio's KaizenThemeProvider populates (native look +
  // dark mode) instead of bundling their own foundations copy.
  '@nvidia/foundations-react-core',
  // Shared so a plugin's useQuery reads Studio's QueryClientProvider — one
  // query cache across Studio and every plugin.
  '@tanstack/react-query',
] as const;

const DEV_VENDOR_DIR = 'node_modules/.vendor';

// Each import specifier in the map resolves to a single vendor bundle.
// 'react/jsx-runtime' shares react.js and 'react-dom/client' shares
// react-dom.js so there's exactly one copy of each package's internals.
const VENDOR_IMPORT_MAP: Record<string, string> = {
  react: 'react.js',
  'react/jsx-runtime': 'react.js',
  'react-dom': 'react-dom.js',
  'react-dom/client': 'react-dom.js',
  'react-router': 'react-router.js',
  '@nvidia/foundations-react-core': 'foundations.js',
  '@tanstack/react-query': 'react-query.js',
  '@nemo/common': 'common.js',
};

// Virtual modules have no filesystem location. Rolldown's resolver falls
// back to the build's cwd for bare specifiers, which is set to projectRoot
// below, so returning the virtual id as-is lets node-resolve find 'react'
// et al. in studio's node_modules.
const virtualShimPlugin = (shims: Record<string, string>): RolldownPlugin => ({
  name: 'virtual-shim',
  resolveId(id) {
    return id in shims ? id : null;
  },
  load(id) {
    return shims[id] ?? null;
  },
});

const configDir = path.dirname(fileURLToPath(import.meta.url));

// The raw rolldown vendor build doesn't read tsconfig paths, so map by hand.
const WORKSPACE_ALIASES: Record<string, string> = {
  '@nemo/common': path.resolve(configDir, '../common'),
  '@nemo/sdk': path.resolve(configDir, '../sdk'),
};

const SOURCE_EXTENSIONS = ['.ts', '.tsx', '.js', '.jsx'];

const IN_NODE_MODULES = /(^|[\\/])node_modules([\\/]|$)/;

const resolveSourceFile = (base: string): string | null => {
  if (fs.existsSync(base) && fs.statSync(base).isFile()) return base;
  for (const ext of SOURCE_EXTENSIONS) {
    if (fs.existsSync(base + ext)) return base + ext;
  }
  for (const ext of SOURCE_EXTENSIONS) {
    const indexFile = path.join(base, `index${ext}`);
    if (fs.existsSync(indexFile)) return indexFile;
  }
  return null;
};

const workspaceSourcePlugin = (): RolldownPlugin => ({
  name: 'workspace-source',
  resolveId(id, importer) {
    // Studio already bundles these same stylesheets through its own graph.
    if (id.endsWith('.css')) return { id: 'virtual:empty-css', external: false };
    const alias = Object.keys(WORKSPACE_ALIASES).find(
      (name) => id === name || id.startsWith(`${name}/`)
    );
    if (alias) {
      const rest = id.slice(alias.length).replace(/^\//, '');
      return resolveSourceFile(path.join(WORKSPACE_ALIASES[alias], rest));
    }
    if (
      importer &&
      !IN_NODE_MODULES.test(importer) &&
      (id.startsWith('./') || id.startsWith('../'))
    ) {
      return resolveSourceFile(path.resolve(path.dirname(importer), id));
    }
    return null;
  },
  load(id) {
    return id === 'virtual:empty-css' ? '' : null;
  },
});

// React's CJS uses `module.exports = require(...)` double-indirection that
// static CJS analysis can't split into named ESM exports, so `export *`
// only catches `default`. We introspect the real module at build time and
// emit explicit re-exports (`export var useState = _react["useState"]`) —
// rolldown treats those as static named exports.
function cjsNamedReexports(alias: string, keys: string[]): string[] {
  return keys.map((k) => `export var ${k} = ${alias}["${k}"];`);
}

// When a CJS dependency does `require("react")` and 'react' is external,
// rolldown leaves the call as `require(...)` guarded by
// `typeof require !== "undefined" ? require : <fallback>`. In the browser
// no `require` exists, so the fallback throws. This banner declares a
// module-scope `require` backed by the ESM imports of the external names,
// so the guarded call sees a real function and returns the shared
// instance from the import map.
function buildRequireShim(externals: readonly string[]): string {
  const imports = externals
    .map((n, i) => `import * as __ext${i} from ${JSON.stringify(n)};`)
    .join(' ');
  const entries = externals.map((n, i) => `${JSON.stringify(n)}:__ext${i}`).join(',');
  // Return the full ESM namespace, not `default`. The namespace carries
  // both `default` and every named export (e.g. react-dom/client's
  // createRoot / hydrateRoot), which CJS consumers expect to find directly
  // on the value returned by require().
  return (
    `${imports} ` +
    `var __externals = {${entries}}; ` +
    'var require = function(s) { ' +
    'var m = __externals[s]; if (m) return m; ' +
    "throw new Error('Dynamic require of \"' + s + '\" is not supported'); " +
    '};'
  );
}

async function buildVendorBundles(
  outdir: string,
  projectRoot: string,
  dev: boolean,
  env: Readonly<Record<string, unknown>>
): Promise<void> {
  // Load React modules from studio's node_modules and enumerate their real
  // runtime exports so we can generate explicit named re-exports.
  const requireFromStudio = createRequire(path.resolve(projectRoot, 'package.json'));
  const keysOf = (spec: string): string[] =>
    Object.keys(requireFromStudio(spec)).filter((k) => k !== '__esModule' && k !== 'default');

  const reactKeys = keysOf('react');
  // jsx-runtime's Fragment overlaps with react's; keep only unique keys.
  const jsxRuntimeKeys = keysOf('react/jsx-runtime').filter((k) => !reactKeys.includes(k));
  // react-dom/client re-declares createRoot/hydrateRoot with
  // `usingClientEntryPoint = true` to silence the legacy-root dev warning,
  // so we export those *from* the client entry and skip them in react-dom's
  // key list to avoid duplicate exports.
  const reactDomClientKeys = keysOf('react-dom/client');
  const reactDomKeys = keysOf('react-dom').filter((k) => !reactDomClientKeys.includes(k));

  const shims: Record<string, string> = {
    'virtual:react': [
      "import _react from 'react';",
      'export default _react;',
      ...cjsNamedReexports('_react', reactKeys),
      "import _jsxRuntime from 'react/jsx-runtime';",
      ...cjsNamedReexports('_jsxRuntime', jsxRuntimeKeys),
    ].join('\n'),
    'virtual:react-dom': [
      "import _reactDom from 'react-dom';",
      "import _reactDomClient from 'react-dom/client';",
      // Both 'react-dom' and 'react-dom/client' import-map to this file,
      // so the default export must satisfy both shapes: `import X from
      // 'react-dom/client'` expects X.createRoot, while `import X from
      // 'react-dom'` expects X.createPortal/flushSync/etc. Merging makes
      // the same object usable from either import style.
      'var _reactDomMerged = Object.assign({}, _reactDom, _reactDomClient);',
      'export default _reactDomMerged;',
      ...cjsNamedReexports('_reactDom', reactDomKeys),
      ...cjsNamedReexports('_reactDomClient', reactDomClientKeys),
    ].join('\n'),
    'virtual:react-router': "export * from 'react-router';",
    // Foundations is ESM with static named exports, so a plain re-export works
    // (no CJS named-reexport introspection needed as with react/react-dom).
    'virtual:foundations': "export * from '@nvidia/foundations-react-core';",
    'virtual:react-query': "export * from '@tanstack/react-query';",
    'virtual:common': "export * from '@nemo/common/src/plugin';",
  };

  const entries: Array<{
    entry: string;
    outfile: string;
    external: string[];
    banner?: string;
    plugins?: RolldownPlugin[];
  }> = [
    { entry: 'virtual:react', outfile: 'react.js', external: [] },
    {
      entry: 'virtual:react-dom',
      outfile: 'react-dom.js',
      external: ['react'],
      banner: buildRequireShim(['react']),
    },
    { entry: 'virtual:react-router', outfile: 'react-router.js', external: ['react'] },
    {
      entry: 'virtual:foundations',
      outfile: 'foundations.js',
      external: ['react', 'react-dom'],
      // Bundled CJS deps inside foundations may `require('react')` /
      // `require('react-dom')`; the shim routes those to the shared copies.
      banner: buildRequireShim(['react', 'react-dom']),
    },
    {
      entry: 'virtual:react-query',
      outfile: 'react-query.js',
      external: ['react'],
      banner: buildRequireShim(['react']),
    },
    {
      entry: 'virtual:common',
      outfile: 'common.js',
      external: [...VENDOR_EXTERNALS],
      banner: buildRequireShim(VENDOR_EXTERNALS),
      plugins: [workspaceSourcePlugin()],
    },
  ];

  fs.rmSync(outdir, { recursive: true, force: true });
  fs.mkdirSync(outdir, { recursive: true });
  await Promise.all(
    entries.map(({ entry, outfile, external, banner, plugins = [] }) =>
      rolldownBuild({
        input: entry,
        cwd: projectRoot,
        platform: 'browser',
        // Dev needs React's development build: Fast Refresh's scheduleRefresh
        // hook and dev warnings only exist in the development renderer.
        transform: {
          define: {
            'process.env.NODE_ENV': dev ? '"development"' : '"production"',
            // Rolldown leaves `import.meta.env` undefined, so shared code
            // reading env vars (PLATFORM_BASE_URL) throws when a plugin loads
            // this bundle. Inline Vite's resolved env instead.
            'import.meta.env': JSON.stringify(env),
          },
        },
        external,
        plugins: [virtualShimPlugin(shims), ...plugins],
        output: {
          file: path.resolve(outdir, outfile),
          format: 'esm',
          // Single-file output per vendor bundle so each maps to one import-map
          // entry. Required for foundations, whose internal dynamic imports
          // would otherwise split into multiple chunks (rejected by output.file).
          codeSplitting: false,
          minify: !dev,
          sourcemap: true,
          banner,
        },
      })
    )
  );
}

// Generates public/vendor/*.js and injects an inline
// <script type="importmap"> into index.html. Inline (not external) because
// external import maps aren't universally supported, and inline resolves
// before any module script — no race window where a dynamically-loaded
// plugin's bare 'react' import can miss the map.
function vendorPlugin(): Plugin {
  let base = '/';
  let outdir = '';
  let projectRoot = '';
  let isServe = false;
  let isPreview = false;
  let viteEnv: Readonly<Record<string, unknown>> = {};
  let pending: Promise<void> | null = null;

  // buildVendorBundles wipes outdir and rebuilds on every vite process start,
  // so dev and build never serve bundles left over from the other mode.
  const ensureBuilt = () => {
    if (!pending) {
      pending = buildVendorBundles(outdir, projectRoot, isServe, viteEnv);
    }
    return pending;
  };

  // Base-aware absolute URLs: entries in the import map are resolved against
  // the document base URL, which on SPA-fallback deep links can be anywhere
  // inside the app, so relative './vendor/…' paths would break. Absolute
  // paths prefixed with Vite's `base` resolve identically from any route.
  const vendorUrl = (bundle: string) =>
    isServe ? `${base}${DEV_VENDOR_DIR}/${bundle}` : `${base}vendor/${bundle}`;

  const importMapJson = () =>
    JSON.stringify({
      imports: Object.fromEntries(
        Object.entries(VENDOR_IMPORT_MAP).map(([k, v]) => [k, vendorUrl(v)])
      ),
    });

  return {
    name: 'studio-plugin-vendor',
    // Run before Vite's default resolver so the dev-mode rewrite below
    // beats optimizeDeps. In build mode rolldownOptions.external handles
    // the same specifiers; resolveId is a no-op there.
    enforce: 'pre',
    // `command` is 'serve' under `vite preview` too, and isPreview is only on
    // ConfigEnv — preview serves an existing dist, so it must not rebuild.
    config(_, env) {
      isPreview = env.isPreview === true;
    },
    async configResolved(config) {
      base = config.base;
      isServe = config.command === 'serve' && !isPreview;
      projectRoot = config.root;
      viteEnv = config.env;
      outdir = isServe
        ? path.resolve(projectRoot, DEV_VENDOR_DIR)
        : path.resolve(projectRoot, 'public', 'vendor');
      if (!isPreview) {
        await ensureBuilt();
      }
    },
    buildStart() {
      return ensureBuilt();
    },
    configureServer() {
      return ensureBuilt();
    },
    // Without this, Vite pre-bundles bare 'react' into .vite/deps and
    // rewrites Studio's imports to that URL, giving Studio its own React
    // while plugins use the vendor copy. Resolving to the same file the
    // dev import map points at keeps one shared instance at runtime.
    resolveId(id) {
      if (!isServe) return null;
      const mapped = VENDOR_IMPORT_MAP[id];
      if (mapped) {
        return path.resolve(outdir, mapped);
      }
      return null;
    },
    transformIndexHtml() {
      return [
        {
          tag: 'script',
          attrs: { type: 'importmap' },
          children: importMapJson(),
          injectTo: 'head-prepend',
        },
      ];
    },
  };
}

const isCI = Boolean(process.env.CI);
const isProd = process.env.NODE_ENV === 'production';
const isTest = Boolean(process.env.VITEST);

const commonPackageDir = path.resolve(configDir, '../common');
const licenseFileNames = [
  'LICENSE',
  'LICENSE.md',
  'LICENSE.txt',
  'LICENCE',
  'LICENCE.md',
  'LICENCE.txt',
  'COPYING',
  'COPYING.md',
  'COPYING.txt',
];

const readPackageJson = (packageJsonPath: string): PackageJson =>
  JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8')) as PackageJson;

const getPackageLicense = (packageJson: PackageJson): string | null => {
  if (typeof packageJson.license === 'string') return packageJson.license;
  if (packageJson.license?.type) return packageJson.license.type;
  return null;
};

const getPackageAuthor = (packageJson: PackageJson): string | null => {
  if (typeof packageJson.author === 'string') return packageJson.author;
  return packageJson.author?.name ?? null;
};

const readLicenseText = (packageDir: string): string | null => {
  const licenseFile = licenseFileNames.find((fileName) =>
    fs.existsSync(path.join(packageDir, fileName))
  );

  return licenseFile ? fs.readFileSync(path.join(packageDir, licenseFile), 'utf-8') : null;
};

const getInstalledPackageDir = (packageName: string, consumerPackageDir: string): string => {
  const packageJsonPath = path.join(
    consumerPackageDir,
    'node_modules',
    packageName,
    'package.json'
  );

  if (!fs.existsSync(packageJsonPath)) {
    throw new Error(`Unable to resolve ${packageName} from ${consumerPackageDir}`);
  }

  return path.dirname(fs.realpathSync(packageJsonPath));
};

// The plugin only reports packages it sees in the Studio bundle graph. Include
// Common's runtime dependencies so the Studio license artifact covers shared UI
// code that may be distributed before every component is imported by Studio.
const getCommonRuntimeLicenseDependencies = (): LicenseReportDependency[] => {
  const commonPackageJson = readPackageJson(path.join(commonPackageDir, 'package.json'));
  const dependencyNames = Object.keys(commonPackageJson.dependencies ?? {})
    .filter((dependencyName) => !dependencyName.startsWith('@nemo/'))
    .sort((first, second) => first.localeCompare(second));

  return dependencyNames.map((dependencyName) => {
    const dependencyDir = getInstalledPackageDir(dependencyName, commonPackageDir);
    const dependencyPackageJson = readPackageJson(path.join(dependencyDir, 'package.json'));

    return {
      name: dependencyPackageJson.name ?? dependencyName,
      version: dependencyPackageJson.version ?? null,
      license: getPackageLicense(dependencyPackageJson),
      author: getPackageAuthor(dependencyPackageJson),
      licenseText: readLicenseText(dependencyDir),
    };
  });
};

const getLicenseDependencyKey = (dependency: LicenseReportDependency): string =>
  `${dependency.name ?? 'Unknown'}@${dependency.version ?? 'Unknown'}`;

const mergeLicenseDependencies = (dependencies: Dependency[]): LicenseReportDependency[] => {
  const seen = new Set(dependencies.map(getLicenseDependencyKey));
  const commonRuntimeDependencies = getCommonRuntimeLicenseDependencies().filter((dependency) => {
    const key = getLicenseDependencyKey(dependency);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return [...dependencies, ...commonRuntimeDependencies];
};

const getAuthorName = (author: LicenseReportDependency['author']): string => {
  if (typeof author === 'string') return author;
  return author?.name ?? 'Unknown';
};

const formatLicenseDependency = (dependency: LicenseReportDependency): string => {
  const license = dependency.license || 'Unknown';
  const name = dependency.name || 'Unknown';
  const version = dependency.version || 'Unknown';
  const author = getAuthorName(dependency.author);
  const licenseText = dependency.licenseText || '';
  return `${name}@${version}\nLicense: ${license}\nAuthor: ${author}\n${licenseText ? `\n${licenseText}\n` : ''}${'='.repeat(80)}`;
};

const formatLicenseReport = (dependencies: Dependency[]): string =>
  mergeLicenseDependencies(dependencies).map(formatLicenseDependency).join('\n\n');

// https://vitejs.dev/config/
// eslint-disable-next-line import/no-default-export
export default defineConfig(({ mode }) => {
  // Load env file based on mode (e.g., .env.fastapi for --mode fastapi)
  const {
    VITE_BASE_URL,
    VITE_DEV_SERVER_HOST,
    VITE_PLATFORM_BASE_URL,
    VITE_PLATFORM_PROXY_DOMAIN,
  } = loadEnv(mode, './env');
  const devServerHost = VITE_DEV_SERVER_HOST?.trim() || 'localhost';
  // Use VITE_BASE_URL to host the app at a subpath (fast mode)
  const base = ['fastapi'].includes(mode) && VITE_BASE_URL ? `/${VITE_BASE_URL}` : '/';

  // Dev-server proxy: when VITE_PLATFORM_BASE_URL is empty the app issues
  // same-origin `/apis/...` requests, so the browser only ever talks to the
  // (HTTPS) dev server and Vite forwards to the plain-HTTP platform server-side.
  // This avoids Safari's mixed-content block on an HTTPS page calling http://.
  // Gated on VITE_PLATFORM_PROXY_DOMAIN so it stays opt-in per developer.
  const proxyDomain = VITE_PLATFORM_PROXY_DOMAIN?.trim();
  const shouldProxyHelix = VITE_PLATFORM_BASE_URL?.trim() === '' && Boolean(proxyDomain);

  const inTestMode = isTest || mode.includes('test');

  // Skip mkcert in tests/CI: it fetches GitHub API for releases and hits rate limits (403) in CI.
  const plugins = [
    react(),
    ...(inTestMode ? [] : [vendorPlugin()]),
    ...(inTestMode ? [] : [mkcert()]),
    svgr(),
  ];

  return mergeConfig(baseTestConfig, {
    envDir: './env',
    base,
    plugins,
    resolve: {
      tsconfigPaths: true,
    },
    // Vite pre-bundles bare deps by default; excluding the shared vendor
    // packages keeps them out of .vite/deps so our resolveId hook can route
    // them to /vendor/* instead without Vite racing to pre-bundle first.
    optimizeDeps: {
      exclude: [...VENDOR_EXTERNALS],
      include: ['@nvidia/foundations-react-core > react-compiler-runtime'],
    },
    build: {
      rolldownOptions: {
        external: [...VENDOR_EXTERNALS],
        // Some CJS deps (e.g. KUI) call `require('react')` internally. With
        // react external, rolldown emits a guarded `require` call whose
        // browser fallback throws. The banner supplies a real `require` in
        // every chunk that routes to the import-map-resolved ESM copy.
        output: {
          banner: buildRequireShim(VENDOR_EXTERNALS),
        },
        plugins: [
          !isCI ? visualizer({ filename: 'dist/stats.html', gzipSize: true }) : undefined,
          license({
            thirdParty: {
              includePrivate: false,
              output: {
                file: 'dist/LICENSES.txt',
                template: formatLicenseReport,
              },
            },
          }),
        ],
      },
      sourcemap: isProd ? false : true,
    },
    css: {
      postcss: {
        plugins: [
          tailwindPostcss({
            base: '../../packages', // We need to tell postcss to look for code in workspace packages
          }),
        ],
      },
    },
    server: {
      host: devServerHost,
      port: 5173,
      ...(shouldProxyHelix
        ? {
            proxy: {
              '/apis': {
                target: proxyDomain,
                changeOrigin: true,
                secure: false,
              },
            },
          }
        : {}),
    },
    worker: {
      format: 'es',
      plugins: () => [react()],
    },
    test: {
      globalSetup: '@nemo/testing/react/global-setup',
      setupFiles: ['@nemo/testing/react/setup', './vitest.setup.tsx'],
      testTimeout: isCI ? 60000 : 10000,
      hookTimeout: isCI ? 60000 : 10000,
      exclude: ['e2e-tests/**', 'node_modules'],
      coverage: {
        include: ['src/**/*.{js,jsx,ts,tsx}'],
        provider: 'v8',
        // In CI skip 'html' reporter to reduce memory; cobertura + text suffice for MR and logs
        reporter: isCI ? ['cobertura', 'text'] : ['cobertura', 'text', 'html', 'json'],
        reportsDirectory: '.test-reports/coverage',
      },
      outputFile: {
        junit: '.test-reports/junit/junit.xml',
      },
    },
  });
});
