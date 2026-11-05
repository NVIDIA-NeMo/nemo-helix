// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  diffContract,
  hasBlockingIssues,
  PLUGIN_CONTRACT,
  type IssueCode,
  type PluginContract,
  type PluginSidecar,
} from '@nemo/common/src/pluginContract';

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
  bundledPackages: [],
  builtAgainst: { ...STUDIO },
  ...overrides,
});

const diff = (s: PluginSidecar | null) => diffContract(s, CONTRACT, STUDIO);
const codes = (s: PluginSidecar | null) => diff(s).map((i) => i.code);

describe('diffContract', () => {
  it('accepts a clean sidecar and a missing sidecar blocks', () => {
    expect(diff(sidecar()), 'clean sidecar should have no issues').toEqual([]);
    expect(hasBlockingIssues([]), 'no issues must not block').toBe(false);

    const missing = diff(null);
    expect(missing, 'null sidecar yields exactly one issue').toHaveLength(1);
    expect(missing[0].code).toBe('PLUGIN_MANIFEST_MISSING');
    expect(missing[0].severity).toBe('error');
    expect(hasBlockingIssues(missing), 'missing manifest must block').toBe(true);
  });

  it('reports unmapped specifiers and bundled singletons, but not bundled provided/unlisted packages', () => {
    const unmapped = diff(sidecar({ specifiers: ['react', 'lodash'] }));
    expect(unmapped, 'one unmapped issue for lodash').toHaveLength(1);
    expect(unmapped[0].code).toBe('PLUGIN_SPECIFIER_UNMAPPED');
    expect(unmapped[0].severity).toBe('error');
    expect(unmapped[0].subject).toBe('lodash');
    expect(unmapped[0].message).toContain('lodash');

    // Bundled lodash is not a bare import, so it is fine.
    expect(
      diff(sidecar({ bundledPackages: ['lodash'] })),
      'bundling a non-contract package is allowed'
    ).toEqual([]);

    const singleton = diff(sidecar({ bundledPackages: ['react-router'] }));
    expect(singleton, 'bundled singleton yields one issue').toHaveLength(1);
    expect(singleton[0].code).toBe('PLUGIN_SINGLETON_BUNDLED');
    expect(singleton[0].severity).toBe('error');
    expect(singleton[0].subject).toBe('react-router');
    expect(singleton[0].message).toContain('react-router');

    expect(
      diff(sidecar({ bundledPackages: ['@nemo/common'] })),
      'bundling a provided package is allowed'
    ).toEqual([]);
  });

  it('applies the contract version policy', () => {
    const unsupported = ['2.0.0', '0.9.0', '1.3.0', '1.10.0'];
    for (const contractVersion of unsupported) {
      expect(codes(sidecar({ contractVersion })), `${contractVersion} vs 1.2.0`).toEqual([
        'PLUGIN_CONTRACT_UNSUPPORTED',
      ]);
      expect(diff(sidecar({ contractVersion }))[0].severity).toBe('error');
    }
    for (const contractVersion of ['1.2.0', '1.2.9', '1.1.0', '1.0.0']) {
      expect(codes(sidecar({ contractVersion })), `${contractVersion} vs 1.2.0`).toEqual([]);
    }
  });

  it('classifies version drift as warning and major mismatch as error', () => {
    const cases: [string, string, IssueCode | null][] = [
      ['8.3.0', '8.3.0', null],
      ['^8.3.0', '8.3.0', null],
      ['~8.3.0', '^8.3.0', null],
      ['8.2.0', '8.3.0', 'PLUGIN_VERSION_DRIFT'],
      ['8.3.1', '8.3.0', 'PLUGIN_VERSION_DRIFT'],
      ['^8.1.5', '8.3.0', 'PLUGIN_VERSION_DRIFT'],
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
      if (expected === 'PLUGIN_VERSION_DRIFT') {
        expect(actual[0].severity, label).toBe('warning');
        expect(hasBlockingIssues(actual), `${label}: drift must not block`).toBe(false);
      }
      if (expected === 'PLUGIN_VERSION_INCOMPATIBLE') {
        expect(actual[0].severity, label).toBe('error');
        expect(hasBlockingIssues(actual), `${label}: major mismatch must block`).toBe(true);
      }
    }

    expect(
      codes(sidecar({ builtAgainst: {} })),
      'packages without a builtAgainst entry are ignored'
    ).toEqual([]);
  });

  it('reports every problem once, errors before warnings, in a stable order', () => {
    const messy = sidecar({
      contractVersion: '1.9.0',
      specifiers: ['react', 'lodash'],
      bundledPackages: ['react-router'],
      builtAgainst: { react: '19.0.0', 'react-router': '7.0.0' },
    });
    const first = diff(messy);
    expect(first.map((i) => i.code).sort()).toEqual(
      [
        'PLUGIN_CONTRACT_UNSUPPORTED',
        'PLUGIN_SINGLETON_BUNDLED',
        'PLUGIN_SPECIFIER_UNMAPPED',
        'PLUGIN_VERSION_DRIFT',
        'PLUGIN_VERSION_INCOMPATIBLE',
      ].sort()
    );
    const severities = first.map((i) => i.severity);
    expect(severities, 'errors must precede warnings').toEqual(
      [...severities].sort((a, b) => (a === b ? 0 : a === 'error' ? -1 : 1))
    );
    expect(diff(messy), 'same input, same order').toEqual(first);
    expect(hasBlockingIssues(first)).toBe(true);
  });

  it('gives every code a message naming its subject and actionable steps', () => {
    const scenarios: Record<IssueCode, PluginSidecar | null> = {
      PLUGIN_MANIFEST_MISSING: null,
      PLUGIN_SPECIFIER_UNMAPPED: sidecar({ specifiers: ['lodash'] }),
      PLUGIN_SINGLETON_BUNDLED: sidecar({ bundledPackages: ['react-router'] }),
      PLUGIN_CONTRACT_UNSUPPORTED: sidecar({ contractVersion: '3.0.0' }),
      PLUGIN_VERSION_DRIFT: sidecar({ builtAgainst: { ...STUDIO, 'react-router': '8.2.0' } }),
      PLUGIN_VERSION_INCOMPATIBLE: sidecar({
        builtAgainst: { ...STUDIO, 'react-router': '7.0.0' },
      }),
    };
    for (const [code, input] of Object.entries(scenarios) as [IssueCode, PluginSidecar | null][]) {
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

  it('tells the developer which version to use for drift and incompatibility', () => {
    for (const built of ['8.2.0', '7.0.0']) {
      const issues = diffContract(sidecar({ builtAgainst: { 'react-router': built } }), CONTRACT, {
        ...STUDIO,
      });
      expect(issues, `built ${built}`).toHaveLength(1);
      const steps = issues[0].steps.join('\n');
      expect(steps, 'names the Studio version').toContain('8.3.0');
      expect(steps, 'names the package').toContain('react-router');
      expect(steps, 'points at web/package.json').toContain('web/package.json');
      expect(steps.toLowerCase(), 'tells the developer to rebuild').toContain('rebuild');
    }
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
