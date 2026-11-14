// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CONTRACT_DOCS,
  PLUGIN_CONTRACT,
  diffContract,
  type PluginSidecar,
} from '@nemo/common/src/pluginContract';
import type { PluginManifest } from '@studio/plugins/types';
import { fetchPlugins, loadPlugin } from '@studio/plugins/utils';

vi.mock('@studio/plugins/studioVersions', () => ({
  STUDIO_SHARED_VERSIONS: { react: '19.2.7', 'react-router': '8.3.0' },
}));

const STUDIO_VERSIONS = { react: '19.2.7', 'react-router': '8.3.0' };
const BASE = 'http://api.test';
const REPORT_PATH = '/studio/api/plugin-load-errors';

const manifest = (name = 'pr-review'): PluginManifest => ({
  name,
  bundleUrl: `/plugin-ui/${name}/index.js`,
});
const sidecarUrl = (name = 'pr-review') => `${BASE}/plugin-ui/${name}/plugin.manifest.json`;
const bundleUrl = (name = 'pr-review') => `${BASE}/plugin-ui/${name}/index.js`;

const goodSidecar = (over: Partial<PluginSidecar> = {}): PluginSidecar => ({
  contractVersion: '1.0.0',
  specifiers: ['react', 'react-router'],
  builtAgainst: { react: '19.2.7', 'react-router': '8.3.0' },
  ...over,
});

const json = (body: unknown, status = 200) =>
  ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  }) as Response;

const urlOf = (input: unknown) => String(input);
const callsTo = (suffix: string) =>
  vi.mocked(global.fetch).mock.calls.filter(([url]) => urlOf(url).endsWith(suffix));

/** Passes the loader's shape check (right containers) but holds non-string values. */
const MALFORMED_SIDECARS: [string, unknown][] = [
  ['a numeric builtAgainst value', { builtAgainst: { react: 19 } }],
  ['an array builtAgainst value', { builtAgainst: { react: ['19.2.7'] } }],
  ['an object builtAgainst value', { builtAgainst: { react: { version: '19.2.7' } } }],
  ['a null builtAgainst value', { builtAgainst: { react: null } }],
  ['a numeric specifier', { specifiers: [1] }],
];

/** A fetch that never answers; like real fetch, it rejects with the signal's reason on abort. */
const hangUntilAborted = (init?: RequestInit) =>
  new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () => reject(init.signal?.reason));
  });

const SIDECAR_TIMEOUT_MS = 10_000;
const DOCS_STEP = `Contract and shared-dependency rules: ${CONTRACT_DOCS}`;

const goodModule = { Root: () => null, navItems: () => [] };

beforeEach(() => {
  vi.resetAllMocks();
  global.fetch = vi.fn();
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

describe('loadPlugin', () => {
  it('returns null for a headless plugin without fetching or importing', async () => {
    const importModule = vi.fn();

    const result = await loadPlugin({ name: 'headless', bundleUrl: null }, BASE, importModule);

    expect(result).toBeNull();
    expect(global.fetch).not.toHaveBeenCalled();
    expect(importModule).not.toHaveBeenCalled();
  });

  it('fails an untrusted bundle URL without fetching or importing', async () => {
    const importModule = vi.fn();

    const result = await loadPlugin(
      { name: 'evil', bundleUrl: 'https://evil.com/malicious.js' },
      BASE,
      importModule
    );

    expect(result).toMatchObject({
      failure: { name: 'evil', issues: [{ code: 'PLUGIN_LOAD_FAILED' }] },
    });
    expect(global.fetch).not.toHaveBeenCalled();
    expect(importModule).not.toHaveBeenCalled();
  });

  it('checks the sidecar once, then imports the bundle and returns the plugin', async () => {
    vi.mocked(global.fetch).mockResolvedValue(json(goodSidecar()));
    const importModule = vi.fn().mockResolvedValue(goodModule);

    const result = await loadPlugin(manifest(), BASE, importModule);

    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(urlOf(vi.mocked(global.fetch).mock.calls[0][0])).toBe(sidecarUrl());
    expect(importModule).toHaveBeenCalledTimes(1);
    expect(importModule).toHaveBeenCalledWith(bundleUrl());
    expect(result).toEqual({
      plugin: { name: 'pr-review', Root: goodModule.Root, navItems: goodModule.navItems },
    });
  });

  it('loads a plugin built against a different minor/patch of the same major', async () => {
    vi.mocked(global.fetch).mockResolvedValue(
      json(goodSidecar({ builtAgainst: { react: '19.1.0' } }))
    );
    const importModule = vi.fn().mockResolvedValue(goodModule);

    const result = await loadPlugin(manifest(), BASE, importModule);

    expect(callsTo('plugin.manifest.json')).toHaveLength(1);
    expect(result).toMatchObject({ plugin: { name: 'pr-review' } });
    expect(importModule).toHaveBeenCalledTimes(1);
  });

  it('reports PLUGIN_MANIFEST_MISSING when the sidecar is a 404, and does not import', async () => {
    vi.mocked(global.fetch).mockResolvedValue(json({}, 404));
    const importModule = vi.fn();

    const result = await loadPlugin(manifest(), BASE, importModule);

    expect(result).toMatchObject({
      failure: { name: 'pr-review', issues: [{ code: 'PLUGIN_MANIFEST_MISSING' }] },
    });
    expect(importModule).not.toHaveBeenCalled();
  });

  it.each<[string, Partial<PluginSidecar>]>([
    ['PLUGIN_SPECIFIER_UNMAPPED', { specifiers: ['react', 'lodash'] }],
    ['PLUGIN_CONTRACT_UNSUPPORTED', { contractVersion: '2.0.0' }],
    ['PLUGIN_VERSION_INCOMPATIBLE', { builtAgainst: { react: '18.3.1' } }],
  ])('passes %s through and blocks the import', async (code, over) => {
    const sidecar = goodSidecar(over);
    vi.mocked(global.fetch).mockResolvedValue(json(sidecar));
    const importModule = vi.fn();
    const expected = diffContract(sidecar, PLUGIN_CONTRACT, STUDIO_VERSIONS);

    const result = await loadPlugin(manifest(), BASE, importModule);

    expect(expected.map((i) => i.code)).toEqual([code]);
    expect(result).toEqual({ failure: { name: 'pr-review', issues: expected } });
    expect(importModule).not.toHaveBeenCalled();
  });

  it('loads a plugin whose sidecar has no bundledPackages field', async () => {
    vi.mocked(global.fetch).mockResolvedValue(json(goodSidecar()));
    const importModule = vi.fn().mockResolvedValue(goodModule);

    const result = await loadPlugin(manifest(), BASE, importModule);

    expect(result).toMatchObject({ plugin: { name: 'pr-review' } });
    expect(importModule).toHaveBeenCalledTimes(1);
  });

  it('ignores unknown sidecar keys such as a legacy bundledPackages', async () => {
    vi.mocked(global.fetch).mockResolvedValue(
      json({ ...goodSidecar(), bundledPackages: ['x'], somethingNew: 1 })
    );
    const importModule = vi.fn().mockResolvedValue(goodModule);

    const result = await loadPlugin(manifest(), BASE, importModule);

    expect(result).toMatchObject({ plugin: { name: 'pr-review' } });
    expect(importModule).toHaveBeenCalledTimes(1);
  });

  describe('sidecar request never answers', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('gives up after 10 seconds with a timed-out PLUGIN_LOAD_FAILED and does not import', async () => {
      vi.mocked(global.fetch).mockImplementation((_url, init) => hangUntilAborted(init));
      const importModule = vi.fn();
      let result: Awaited<ReturnType<typeof loadPlugin>> | undefined;
      void loadPlugin(manifest(), BASE, importModule).then((r) => {
        result = r;
      });

      await vi.advanceTimersByTimeAsync(SIDECAR_TIMEOUT_MS);

      expect(vi.mocked(global.fetch).mock.calls[0][1]?.signal).toBeInstanceOf(AbortSignal);
      expect(result).toMatchObject({
        failure: {
          name: 'pr-review',
          issues: [
            { code: 'PLUGIN_LOAD_FAILED', message: expect.stringMatching(/timed out|timeout/i) },
          ],
        },
      });
      expect(importModule).not.toHaveBeenCalled();
    });
  });

  describe('generic load failures', () => {
    it.each<[string, () => Promise<Awaited<ReturnType<typeof loadPlugin>>>]>([
      [
        'an untrusted bundle URL',
        () => loadPlugin({ name: 'evil', bundleUrl: 'https://evil.com/x.js' }, BASE, vi.fn()),
      ],
      [
        'a bundle that fails to import',
        () => {
          vi.mocked(global.fetch).mockResolvedValue(json(goodSidecar()));
          return loadPlugin(manifest(), BASE, vi.fn().mockRejectedValue(new Error('nope')));
        },
      ],
    ])('ends the steps with the contract docs pointer for %s', async (_label, run) => {
      const result = await run();

      if (!result || !('failure' in result)) throw new Error('expected a failure result');
      expect(result.failure.issues[0].code).toBe('PLUGIN_LOAD_FAILED');
      expect(result.failure.issues[0].steps.at(-1)).toBe(DOCS_STEP);
    });
  });

  describe('sidecar cannot be read', () => {
    it.each([
      ['the request rejects', () => vi.mocked(global.fetch).mockRejectedValue(new Error('boom'))],
      ['the server answers 500', () => vi.mocked(global.fetch).mockResolvedValue(json({}, 500))],
      [
        'the body is not JSON',
        () =>
          vi.mocked(global.fetch).mockResolvedValue({
            ok: true,
            status: 200,
            json: async () => {
              throw new SyntaxError('Unexpected token <');
            },
            text: async () => '<html>',
          } as unknown as Response),
      ],
      ['the JSON has the wrong shape', () => vi.mocked(global.fetch).mockResolvedValue(json({}))],
    ])('fails with PLUGIN_LOAD_FAILED when %s', async (_label, arrange) => {
      arrange();
      const importModule = vi.fn();

      const result = await loadPlugin(manifest(), BASE, importModule);

      expect(result).toMatchObject({
        failure: {
          name: 'pr-review',
          issues: [{ code: 'PLUGIN_LOAD_FAILED', message: expect.stringMatching(/\S/) }],
        },
      });
      expect(importModule).not.toHaveBeenCalled();
    });
  });

  describe('sidecar has the right shape but malformed values', () => {
    it.each(MALFORMED_SIDECARS)(
      'resolves to PLUGIN_LOAD_FAILED for %s and does not import',
      async (_label, over) => {
        vi.mocked(global.fetch).mockResolvedValue(json({ ...goodSidecar(), ...(over as object) }));
        const importModule = vi.fn();

        const result = await loadPlugin(manifest(), BASE, importModule);

        expect(result).toMatchObject({
          failure: {
            name: 'pr-review',
            issues: [
              {
                code: 'PLUGIN_LOAD_FAILED',
                message: expect.stringMatching(/not (a )?valid|invalid/i),
              },
            ],
          },
        });
        expect(importModule).not.toHaveBeenCalled();
      }
    );
  });

  describe('bundle import', () => {
    beforeEach(() => {
      vi.mocked(global.fetch).mockResolvedValue(json(goodSidecar()));
    });

    it('carries the import error message and fix steps into a PLUGIN_LOAD_FAILED failure', async () => {
      const importModule = vi
        .fn()
        .mockRejectedValue(new Error('Failed to resolve module specifier "lodash"'));

      const result = await loadPlugin(manifest(), BASE, importModule);

      expect(result).toMatchObject({
        failure: {
          name: 'pr-review',
          issues: [
            {
              code: 'PLUGIN_LOAD_FAILED',
              message: expect.stringContaining('Failed to resolve module specifier "lodash"'),
              steps: expect.arrayContaining([expect.any(String)]),
            },
          ],
        },
      });
    });

    it('fails with PLUGIN_LOAD_FAILED when the module lacks Root/navItems', async () => {
      const importModule = vi.fn().mockResolvedValue({ Root: () => null });

      const result = await loadPlugin(manifest(), BASE, importModule);

      expect(result).toMatchObject({
        failure: { name: 'pr-review', issues: [{ code: 'PLUGIN_LOAD_FAILED' }] },
      });
    });
  });
});

describe('fetchPlugins', () => {
  /** Route global.fetch: manifest list, sidecars (404 unless given), and the report POST. */
  const routeFetch = (
    manifests: PluginManifest[],
    opts: {
      sidecar?: (url: string, init?: RequestInit) => Promise<Response>;
      report?: () => Promise<Response>;
    } = {}
  ) => {
    vi.mocked(global.fetch).mockImplementation(async (input, init) => {
      const url = urlOf(input);
      if (url.endsWith('/apis/plugins')) return json(manifests);
      if (url.endsWith('plugin.manifest.json')) return opts.sidecar?.(url, init) ?? json({}, 404);
      if (url.endsWith(REPORT_PATH)) return opts.report?.() ?? json({}, 202);
      throw new Error(`unexpected fetch ${url}`);
    });
  };

  // The report is fire-and-forget behind async auth plumbing; one macrotask lets it (and any
  // unhandled rejection) surface. Needed for "no POST" assertions, which have nothing to await.
  const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

  it('returns failed and installed plugins alongside loaded ones', async () => {
    routeFetch([{ name: 'headless', bundleUrl: null }, manifest()]);

    const result = await fetchPlugins();

    expect(result.plugins).toEqual([]);
    expect(result.failed).toHaveLength(1);
    expect(result.failed[0].name).toBe('pr-review');
    expect(result.failed[0].issues.map((i) => i.code)).toEqual(['PLUGIN_MANIFEST_MISSING']);
    expect([...result.installedNames].sort()).toEqual(['headless', 'pr-review']);
  });

  it('posts one report per failed plugin', async () => {
    routeFetch([manifest(), { name: 'evil', bundleUrl: 'https://evil.com/x.js' }]);
    const [missing] = diffContract(null, PLUGIN_CONTRACT, STUDIO_VERSIONS);

    await fetchPlugins();
    await flush();

    const posts = callsTo(REPORT_PATH);
    expect(posts).toHaveLength(2);
    const [, init] = posts.find(([, i]) => JSON.parse(String(i?.body)).plugin === 'pr-review')!;
    expect(init?.method).toBe('POST');
    expect(new Headers(init?.headers).get('content-type')).toContain('application/json');
    expect(JSON.parse(String(init?.body))).toEqual({
      plugin: 'pr-review',
      issues: [
        {
          code: 'PLUGIN_MANIFEST_MISSING',
          subject: missing.subject,
          message: missing.message,
          steps: missing.steps,
        },
      ],
    });
  });

  it('does not post when nothing failed', async () => {
    routeFetch([{ name: 'headless', bundleUrl: null }]);

    await fetchPlugins();
    await flush();

    expect(callsTo(REPORT_PATH)).toHaveLength(0);
  });

  it.each([
    ['answers 404', () => Promise.resolve(json({}, 404))],
    ['answers 500', () => Promise.resolve(json({}, 500))],
    ['rejects', () => Promise.reject(new Error('offline'))],
  ])('is unaffected when the report POST %s', async (_label, report) => {
    routeFetch([manifest()], { report });

    const result = await fetchPlugins();
    await flush();

    expect(callsTo(REPORT_PATH)).toHaveLength(1);
    expect(result.failed.map((f) => f.name)).toEqual(['pr-review']);
    expect(result.plugins).toEqual([]);
  });

  it('does not wait for the report POST', async () => {
    routeFetch([manifest()], { report: () => new Promise<Response>(() => {}) });

    const result = await fetchPlugins();

    expect(result.failed).toHaveLength(1);
  });

  describe('a plugin with a malformed sidecar', () => {
    const malformed = json({ ...goodSidecar(), builtAgainst: { react: 19 } });

    it('does not stop a headless plugin from being accounted for', async () => {
      routeFetch([manifest('bad'), { name: 'headless', bundleUrl: null }], {
        sidecar: () => Promise.resolve(malformed),
      });

      const result = await fetchPlugins();

      expect(result.failed.map((f) => [f.name, f.issues.map((i) => i.code)])).toEqual([
        ['bad', ['PLUGIN_LOAD_FAILED']],
      ]);
      expect([...result.installedNames].sort()).toEqual(['bad', 'headless']);
      expect(result.plugins).toEqual([]);
    });

    it('does not stop a plugin with a missing sidecar from being reported', async () => {
      routeFetch([manifest('bad'), manifest('nomanifest')], {
        sidecar: (url) => Promise.resolve(url.includes('/bad/') ? malformed : json({}, 404)),
      });

      const result = await fetchPlugins();
      await flush();

      const codesByName = Object.fromEntries(
        result.failed.map((f) => [f.name, f.issues.map((i) => i.code)])
      );
      expect(codesByName).toEqual({
        bad: ['PLUGIN_LOAD_FAILED'],
        nomanifest: ['PLUGIN_MANIFEST_MISSING'],
      });
      expect(callsTo(REPORT_PATH)).toHaveLength(2);
    });
  });

  describe('a plugin whose sidecar request never answers', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('is reported as failed after 10 seconds without blocking the other plugins', async () => {
      routeFetch([manifest('hung'), { name: 'headless', bundleUrl: null }], {
        sidecar: (_url, init) => hangUntilAborted(init),
      });
      let result: Awaited<ReturnType<typeof fetchPlugins>> | undefined;
      void fetchPlugins().then((r) => {
        result = r;
      });

      await vi.advanceTimersByTimeAsync(0);
      expect(result).toBeUndefined();
      await vi.advanceTimersByTimeAsync(SIDECAR_TIMEOUT_MS);

      expect(result?.failed.map((f) => [f.name, f.issues.map((i) => i.code)])).toEqual([
        ['hung', ['PLUGIN_LOAD_FAILED']],
      ]);
      expect([...(result?.installedNames ?? [])].sort()).toEqual(['headless', 'hung']);
      expect(callsTo(REPORT_PATH)).toHaveLength(1);
    });
  });

  describe('report clamping', () => {
    const postedIssues = async () => {
      await fetchPlugins();
      await flush();
      const posts = callsTo(REPORT_PATH);
      expect(posts).toHaveLength(1);
      const [, init] = posts[0];
      return JSON.parse(String(init?.body)).issues as {
        code: string;
        subject: string;
        message: string;
        steps: string[];
      }[];
    };

    it('cuts a long message to 500 chars, keeping the prefix', async () => {
      const long = 'x'.repeat(600);
      routeFetch([manifest()], { sidecar: () => Promise.reject(new Error(long)) });

      const [issue] = await postedIssues();

      expect(issue.code).toBe('PLUGIN_LOAD_FAILED');
      expect(issue.message.length).toBeLessThanOrEqual(500);
      expect(issue.message.length).toBeGreaterThan(0);
      // The reason is in the message, so some of it must survive the cut.
      expect(issue.message).toContain('xxx');
    });

    it('caps issues at 10 and subject/message/step lengths, keeping prefixes', async () => {
      // 12 unmapped specifiers, each 400 chars: 12 issues whose subject, message and first
      // step all exceed their limits (the message and steps quote the specifier).
      const specifiers = Array.from(
        { length: 12 },
        (_, i) => `s${String(i).padStart(2, '0')}${'x'.repeat(397)}`
      );
      const sidecar = goodSidecar({ specifiers });
      const expected = diffContract(sidecar, PLUGIN_CONTRACT, STUDIO_VERSIONS);
      // Preconditions: every issue really is over each limit (otherwise the cuts prove nothing).
      expect(expected).toHaveLength(12);
      for (const e of expected) {
        expect(e.subject.length).toBeGreaterThan(200);
        expect(e.message.length).toBeGreaterThan(500);
        expect(e.steps[0].length).toBeGreaterThan(300);
      }
      routeFetch([manifest()], { sidecar: () => Promise.resolve(json(sidecar)) });

      const issues = await postedIssues();

      expect(issues).toHaveLength(10);
      issues.forEach((posted, i) => {
        expect(posted.code).toBe(expected[i].code);
        expect(posted.subject.length).toBeLessThanOrEqual(200);
        expect(expected[i].subject.startsWith(posted.subject)).toBe(true);
        expect(posted.subject.length).toBeGreaterThan(0);
        expect(posted.message.length).toBeLessThanOrEqual(500);
        expect(expected[i].message.startsWith(posted.message)).toBe(true);
        expect(posted.steps.length).toBeLessThanOrEqual(8);
        posted.steps.forEach((step, s) => {
          expect(step.length).toBeLessThanOrEqual(300);
          expect(expected[i].steps[s].startsWith(step)).toBe(true);
        });
      });
    });
  });
});
