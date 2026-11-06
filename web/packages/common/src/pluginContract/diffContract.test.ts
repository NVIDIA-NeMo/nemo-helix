// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CONTRACT_DOCS,
  diffContract,
  PLUGIN_CONTRACT,
  type IssueCode,
  type PluginContract,
  type PluginSidecar,
} from '@nemo/common/src/pluginContract';
import { singletonBundled } from '@nemo/common/src/pluginContract/remediate';

const CONTRACT: PluginContract = {
  version: '1.2.0',
  entries: [
    { specifier: 'react', package: 'react', kind: 'singleton' },
    { specifier: 'react/jsx-runtime', package: 'react', kind: 'singleton' },
    { specifier: 'react-router', package: 'react-router', kind: 'singleton' },
    { specifier: '@nemo/common', package: '@nemo/common', kind: 'provided' },
  ],
};

const STUDIO = { react: '19.1.0', 'react-router': '8.3.0', '@nemo/common': '1.0.0' };

const sidecar = (overrides: Partial<PluginSidecar> = {}): PluginSidecar => ({
  contractVersion: '1.2.0',
  specifiers: ['react', 'react/jsx-runtime', 'react-router', '@nemo/common'],
  builtAgainst: { ...STUDIO },
  ...overrides,
});

const diff = (s: PluginSidecar | null) => diffContract(s, CONTRACT, STUDIO);
const codes = (s: PluginSidecar | null) => diff(s).map((i) => i.code);

const DIFF_SCENARIOS: Record<
  Exclude<IssueCode, 'PLUGIN_SINGLETON_BUNDLED'>,
  PluginSidecar | null
> = {
  PLUGIN_MANIFEST_MISSING: null,
  PLUGIN_SPECIFIER_UNMAPPED: sidecar({ specifiers: ['lodash'] }),
  PLUGIN_CONTRACT_UNSUPPORTED: sidecar({ contractVersion: '3.0.0' }),
  PLUGIN_VERSION_INCOMPATIBLE: sidecar({
    builtAgainst: { ...STUDIO, 'react-router': '7.0.0' },
  }),
};

describe('diffContract', () => {
  it('accepts a clean sidecar and rejects a missing sidecar', () => {
    expect(diff(sidecar()), 'clean sidecar should have no issues').toEqual([]);

    const missing = diff(null);
    expect(missing, 'null sidecar yields exactly one issue').toHaveLength(1);
    expect(missing[0].code).toBe('PLUGIN_MANIFEST_MISSING');
  });

  it('reports unmapped specifiers', () => {
    const unmapped = diff(sidecar({ specifiers: ['react', 'lodash'] }));
    expect(unmapped, 'one unmapped issue for lodash').toHaveLength(1);
    expect(unmapped[0].code).toBe('PLUGIN_SPECIFIER_UNMAPPED');
    expect(unmapped[0].subject).toBe('lodash');
    expect(unmapped[0].message).toContain('lodash');
  });

  it('ignores a legacy bundledPackages field', () => {
    const legacy = { ...sidecar(), bundledPackages: ['react-router'] } as PluginSidecar;
    expect(diff(legacy), 'the field is ignored, not reported').toEqual([]);
  });

  it('applies the contract version policy', () => {
    const unsupported = ['2.0.0', '0.9.0', '1.3.0', '1.10.0'];
    for (const contractVersion of unsupported) {
      expect(codes(sidecar({ contractVersion })), `${contractVersion} vs 1.2.0`).toEqual([
        'PLUGIN_CONTRACT_UNSUPPORTED',
      ]);
    }
    for (const contractVersion of ['1.2.0', '1.2.9', '1.1.0', '1.0.0']) {
      expect(codes(sidecar({ contractVersion })), `${contractVersion} vs 1.2.0`).toEqual([]);
    }
  });

  it('flags only a different major as incompatible; minor and patch differences are fine', () => {
    const cases: [string, string, IssueCode | null][] = [
      ['8.3.0', '8.3.0', null],
      ['^8.3.0', '8.3.0', null],
      ['~8.3.0', '^8.3.0', null],
      ['8.2.0', '8.3.0', null],
      ['8.3.1', '8.3.0', null],
      ['^8.1.5', '8.3.0', null],
      ['7.9.0', '8.3.0', 'PLUGIN_VERSION_INCOMPATIBLE'],
      ['^9.0.0', '8.3.0', 'PLUGIN_VERSION_INCOMPATIBLE'],
    ];
    for (const [built, shipped, expected] of cases) {
      const label = `built ${built} vs studio ${shipped}`;
      const actual = diffContract(sidecar({ builtAgainst: { 'react-router': built } }), CONTRACT, {
        ...STUDIO,
        'react-router': shipped,
      });
      expect(
        actual.map((i) => i.code),
        label
      ).toEqual(expected ? [expected] : []);
    }

    for (const built of ['19.2.3', '19.1.9', '^19.0.0']) {
      expect(
        codes(sidecar({ builtAgainst: { react: built } })),
        `react built ${built} vs studio 19.1.0 is same-major and must not be reported`
      ).toEqual([]);
    }
    expect(
      codes(sidecar({ builtAgainst: { react: '18.3.1' } })),
      'react built 18.3.1 vs studio 19.1.0'
    ).toEqual(['PLUGIN_VERSION_INCOMPATIBLE']);

    expect(
      codes(sidecar({ builtAgainst: {} })),
      'packages without a builtAgainst entry are ignored'
    ).toEqual([]);
  });

  it('reports every problem once, in a stable order', () => {
    const messy = sidecar({
      contractVersion: '1.9.0',
      specifiers: ['react', 'lodash'],
      builtAgainst: { react: '18.3.1', 'react-router': '7.0.0', '@nemo/common': '1.5.0' },
    });
    const first = diff(messy);
    expect(first.map((i) => i.code).sort()).toEqual(
      [
        'PLUGIN_CONTRACT_UNSUPPORTED',
        'PLUGIN_SPECIFIER_UNMAPPED',
        'PLUGIN_VERSION_INCOMPATIBLE',
        'PLUGIN_VERSION_INCOMPATIBLE',
      ].sort()
    );
    expect(
      first.filter((i) => i.code === 'PLUGIN_VERSION_INCOMPATIBLE').map((i) => i.subject),
      'one incompatible issue per mismatched package, not for same-major differences'
    ).toEqual(expect.arrayContaining(['react', 'react-router']));
    expect(diff(messy), 'same input, same order').toEqual(first);
  });

  it('gives every diffContract code a message naming its subject and actionable steps', () => {
    for (const [code, input] of Object.entries(DIFF_SCENARIOS)) {
      const issue = diff(input).find((i) => i.code === code);
      expect(issue, `${code} should be emitted`).toBeDefined();
      if (code !== 'PLUGIN_MANIFEST_MISSING') {
        expect(issue!.message, `${code} message names its subject`).toContain(issue!.subject);
      }
      expect(issue!.steps.length, `${code} has steps`).toBeGreaterThan(0);
      expect(
        issue!.steps.every((s) => s.trim().length > 0),
        `${code} steps are non-empty`
      ).toBe(true);
    }
  });

  it('gives the build-time singleton-bundled fix a message, steps and the docs step', () => {
    const fix = singletonBundled('react-router');
    expect(fix.message, 'names the package').toContain('react-router');
    expect(fix.steps.length, 'has steps').toBeGreaterThanOrEqual(1);
    expect(fix.steps.at(-1), 'last step is the docs step').toBe(
      `Contract and shared-dependency rules: ${CONTRACT_DOCS}`
    );
  });

  it('tells the developer which version to use for a major mismatch', () => {
    const cases = [
      { pkg: 'react-router', built: '7.0.0', shipped: '8.3.0' },
      { pkg: 'react', built: '18.3.1', shipped: '19.1.0' },
    ];
    for (const { pkg, built, shipped } of cases) {
      const issues = diff(sidecar({ builtAgainst: { [pkg]: built } }));
      expect(
        issues.map((i) => i.code),
        `${pkg} built ${built}`
      ).toEqual(['PLUGIN_VERSION_INCOMPATIBLE']);
      const steps = issues[0].steps.join('\n');
      expect(steps, 'names the Studio version').toContain(shipped);
      expect(steps, 'names the package').toContain(pkg);
      expect(steps, 'points at web/package.json').toContain('web/package.json');
      expect(steps.toLowerCase(), 'tells the developer to rebuild').toContain('rebuild');
    }
  });
});

describe('diffContract hardening', () => {
  it.each(['abc', '', '>=1', '1.2'])('fails closed on unparseable contractVersion %j', (v) => {
    const issues = diff(sidecar({ contractVersion: v }));
    expect(
      issues.map((i) => i.code),
      `contractVersion ${JSON.stringify(v)}`
    ).toEqual(['PLUGIN_CONTRACT_UNSUPPORTED']);
    expect(issues.length, 'must be rejected').toBeGreaterThan(0);
  });

  it('ends every diffContract issue with the contract docs step', () => {
    for (const [code, input] of Object.entries(DIFF_SCENARIOS)) {
      const issue = diff(input).find((i) => i.code === code);
      expect(issue, `${code} should be emitted`).toBeDefined();
      expect(issue!.steps.at(-1), `${code} last step is the docs step`).toBe(
        `Contract and shared-dependency rules: ${CONTRACT_DOCS}`
      );
    }
  });

  it('exports the docs location, naming the helix repo', () => {
    expect(CONTRACT_DOCS).toBe('plugins/example-plugin/web/AGENTS.md (in the nemo-helix repo)');
  });

  it('skips malformed builtAgainst versions and still parses pre-releases', () => {
    // Values that would be a different major if a lenient parser read a prefix of them.
    const malformed = ['19.2.3garbage', '19.2.3.4', '18.3.1garbage', '18.3.1.4', '>=18', '18', ''];
    for (const built of malformed) {
      expect(
        codes(sidecar({ builtAgainst: { react: built } })),
        `built ${JSON.stringify(built)} is malformed and skipped`
      ).toEqual([]);
    }
    expect(
      codes(sidecar({ builtAgainst: { react: '19.1.0-rc.1' } })),
      'pre-release parses as 19.1.0'
    ).toEqual([]);
    expect(
      codes(sidecar({ builtAgainst: { react: '18.0.0-rc.1' } })),
      'pre-release of another major is still a major mismatch'
    ).toEqual(['PLUGIN_VERSION_INCOMPATIBLE']);
  });

  it.each([
    ['leading zero', '01.0.0'],
    ['empty prerelease', '1.0.0-'],
    ['empty build', '1.0.0+'],
    ['beyond MAX_SAFE_INTEGER', `1${'0'.repeat(400)}.0.0`],
  ])('fails closed on a non-strict contractVersion (%s)', (_label, v) => {
    expect(codes(sidecar({ contractVersion: v }))).toEqual(['PLUGIN_CONTRACT_UNSUPPORTED']);
  });

  it('parses built-against versions strictly', () => {
    expect(
      codes(sidecar({ builtAgainst: { react: '01.2.3' } })),
      'leading-zero version is unparseable and skipped'
    ).toEqual([]);
    expect(
      codes(sidecar({ builtAgainst: { react: '19.1.0-rc.1' } })),
      'pre-release equals 19.1.0'
    ).toEqual([]);
    expect(
      codes(sidecar({ builtAgainst: { react: '^19.1.0+build.5' } })),
      'caret + build metadata equals 19.1.0'
    ).toEqual([]);
    expect(
      codes(sidecar({ builtAgainst: { react: '18.0.0+build.5' } })),
      'build metadata of another major is still a major mismatch'
    ).toEqual(['PLUGIN_VERSION_INCOMPATIBLE']);
  });
});

describe('PLUGIN_CONTRACT', () => {
  it('classifies the shared dependencies', () => {
    const kinds = Object.fromEntries(PLUGIN_CONTRACT.entries.map((e) => [e.specifier, e.kind]));
    expect(kinds).toEqual({
      react: 'singleton',
      'react/jsx-runtime': 'singleton',
      'react-dom': 'singleton',
      'react-dom/client': 'singleton',
      'react-router': 'singleton',
      '@nvidia/foundations-react-core': 'singleton',
      '@tanstack/react-query': 'singleton',
      '@nemo/common': 'provided',
    });
    expect(PLUGIN_CONTRACT.entries.every((e) => e.package.length > 0)).toBe(true);
    const specifiers = PLUGIN_CONTRACT.entries.map((e) => e.specifier);
    expect(new Set(specifiers).size, 'specifiers are unique').toBe(specifiers.length);
    expect(PLUGIN_CONTRACT.version).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
