// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  contractUnsupported,
  manifestMissing,
  singletonBundled,
  specifierUnmapped,
  versionDrift,
  versionIncompatible,
} from '@nemo/common/src/pluginContract/remediate';
import { parseVersion } from '@nemo/common/src/pluginContract/semver';
import type {
  ContractIssue,
  IssueCode,
  IssueSeverity,
  PluginContract,
  PluginSidecar,
} from '@nemo/common/src/pluginContract/types';

const SEVERITY: Record<IssueCode, IssueSeverity> = {
  PLUGIN_MANIFEST_MISSING: 'error',
  PLUGIN_SPECIFIER_UNMAPPED: 'error',
  PLUGIN_SINGLETON_BUNDLED: 'error',
  PLUGIN_CONTRACT_UNSUPPORTED: 'error',
  PLUGIN_VERSION_INCOMPATIBLE: 'error',
  PLUGIN_VERSION_DRIFT: 'warning',
};

const issue = (
  code: IssueCode,
  subject: string,
  fix: Pick<ContractIssue, 'message' | 'steps'>
): ContractIssue => ({ code, severity: SEVERITY[code], subject, ...fix });

const bySubject = (a: ContractIssue, b: ContractIssue) =>
  a.subject < b.subject ? -1 : a.subject > b.subject ? 1 : 0;

/**
 * Compare a plugin's sidecar against the contract. `studioVersions` maps each
 * shared package to the version Studio ships. A `null` sidecar means the plugin
 * was built without the Studio plugin build helper.
 */
export const diffContract = (
  sidecar: PluginSidecar | null,
  contract: PluginContract,
  studioVersions: Readonly<Record<string, string>>
): ContractIssue[] => {
  if (!sidecar) {
    return [issue('PLUGIN_MANIFEST_MISSING', '', manifestMissing())];
  }
  const issues: ContractIssue[] = [];

  const built = parseVersion(sidecar.contractVersion);
  const current = parseVersion(contract.version);
  if (!built || (current && (built[0] !== current[0] || built[1] > current[1]))) {
    issues.push(
      issue(
        'PLUGIN_CONTRACT_UNSUPPORTED',
        sidecar.contractVersion,
        contractUnsupported(sidecar.contractVersion, contract.version)
      )
    );
  }

  const known = new Set(contract.entries.map((e) => e.specifier));
  issues.push(
    ...sidecar.specifiers
      .filter((s) => !known.has(s))
      .map((s) => issue('PLUGIN_SPECIFIER_UNMAPPED', s, specifierUnmapped(s)))
      .sort(bySubject)
  );

  const singletons = new Set(
    contract.entries.filter((e) => e.kind === 'singleton').map((e) => e.package)
  );
  issues.push(
    ...sidecar.bundledPackages
      .filter((p) => singletons.has(p))
      .map((p) => issue('PLUGIN_SINGLETON_BUNDLED', p, singletonBundled(p)))
      .sort(bySubject)
  );

  const versionIssues: ContractIssue[] = [];
  for (const [pkg, builtVersion] of Object.entries(sidecar.builtAgainst)) {
    const studio = studioVersions[pkg];
    const b = parseVersion(builtVersion);
    const s = studio === undefined ? null : parseVersion(studio);
    if (!b || !s) continue;
    if (b[0] !== s[0]) {
      versionIssues.push(
        issue('PLUGIN_VERSION_INCOMPATIBLE', pkg, versionIncompatible(pkg, builtVersion, studio))
      );
    } else if (b[1] !== s[1] || b[2] !== s[2]) {
      versionIssues.push(
        issue('PLUGIN_VERSION_DRIFT', pkg, versionDrift(pkg, builtVersion, studio))
      );
    }
  }
  issues.push(...versionIssues.sort(bySubject));

  // Stable sort: errors first, rule order preserved within a severity.
  return [
    ...issues.filter((i) => i.severity === 'error'),
    ...issues.filter((i) => i.severity === 'warning'),
  ];
};

/** True when any issue should stop the plugin from loading. */
export const hasBlockingIssues = (issues: readonly ContractIssue[]): boolean =>
  issues.some((i) => i.severity === 'error');
