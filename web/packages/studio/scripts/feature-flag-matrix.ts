// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable no-console -- CLI script */
/**
 * Feature Flag Matrix
 *
 * Reads feature flag defaults from multiple sources and displays a combined
 * matrix showing the effective value for each deployment environment.
 *
 * Sources:
 *   1. Studio code defaults (featureFlags.ts)
 *   2. Local env overrides (.env.dev.local)
 *   3. env_mappings.py defaults (deployment runtime injection)
 *   4. Optional deployment values files from HELM_DEPLOYMENT_VALUES_DIR
 *
 * Usage:
 *   pnpm feature-flags          # CLI color-coded per-flag cards
 *   pnpm feature-flags --json   # Machine-readable JSON for agents
 */

import yaml from 'js-yaml';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
const FEATURE_FLAGS_TS = path.join(
  REPO_ROOT,
  'web/packages/studio/src/constants/featureFlags/featureFlags.ts'
);
const ENV_MAPPINGS_PY = path.join(REPO_ROOT, 'services/studio/src/nhx/studio/env_mappings.py');
const LOCAL_ENV_FILE = path.join(REPO_ROOT, 'web/packages/studio/env/.env.dev.local');
const HELM_DEPLOYMENT_VALUES_DIR = process.env.HELM_DEPLOYMENT_VALUES_DIR
  ? path.resolve(REPO_ROOT, process.env.HELM_DEPLOYMENT_VALUES_DIR)
  : '';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const readFile = (p: string) => fs.readFileSync(p, 'utf-8');

const snakeToCamel = (s: string) =>
  s.toLowerCase().replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());

/** Derive a deployment target name: "dev-values.yaml" → "DEV" */
const envName = (yamlFile: string) => yamlFile.replace(/-values\.yaml$/, '').toUpperCase();

/** Run a regex globally and collect results via a mapper function */
function matchAll<T>(re: RegExp, src: string, mapper: (m: RegExpExecArray) => T): T[] {
  const results: T[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(src))) results.push(mapper(m));
  return results;
}

// ---------------------------------------------------------------------------
// Parsers
// ---------------------------------------------------------------------------

/**
 * Parse code-level defaults from featureFlags.ts.
 * Looks for: `flagName: previewFlag('VITE_FF_...', defaultValue)` or `booleanFlag(...)`.
 * These are the lowest-priority defaults, used by the local dev server when no
 * .env.dev.local override exists.
 */
function parseStudioDefaults(src: string): Record<string, string> {
  const stripped = src.replace(/\/\/.*$/gm, '');
  const re = /(\w+):\s*(?:previewFlag|booleanFlag)\(\s*'[^']+'\s*(?:,\s*([^)]*))?\)/g;
  const entries = matchAll(re, stripped, (m) => {
    const raw = m[2]?.trim();
    return [m[1], !raw ? 'false' : raw.replace(/['"]/g, '')] as const;
  });
  return Object.fromEntries(entries);
}

/**
 * Parse deployment defaults from env_mappings.py.
 * Looks for EnvMapping entries with marker="STUDIO_UI_VITE_FF_*" and extracts the
 * `default` value. This is the base default for all deployed environments (DEV,
 * INTERNAL, PROD, etc.) before any Helm values yaml overrides are applied.
 */
function parseEnvMappings(src: string): Record<string, string> {
  const re =
    /marker="STUDIO_UI_(VITE_FF_\w+)"[\s\S]*?config_path="studio\.feature_flags\.(\w+)"[\s\S]*?default="([^"]*)"/g;
  const entries = matchAll(re, src, (m) => [snakeToCamel(m[2]), m[3]] as const);
  return Object.fromEntries(entries);
}

/**
 * Parse local developer overrides from .env.dev.local.
 * Looks for `VITE_FF_*=value` lines. These override the featureFlags.ts defaults
 * on the local dev server only. This file is gitignored and per-developer.
 */
function parseLocalEnv(src: string): Record<string, string> {
  const re = /^VITE_FF_(\w+)=(.+)$/gm;
  const trimQuotes = (s: string) => {
    const q = s[0];
    if ((q === "'" || q === '"') && s.endsWith(q)) return s.slice(1, -1);
    return s;
  };
  const entries = matchAll(re, src, (m) => [snakeToCamel(m[1]), trimQuotes(m[2].trim())] as const);
  return Object.fromEntries(entries);
}

/**
 * Parse per-target overrides from a Helm deployment values yaml (e.g. dev-values.yaml).
 * Reads platformConfig.studio.feature_flags and extracts snake_case key/value pairs.
 * These are the highest-priority values for deployed environments, overriding
 * env_mappings.py defaults for that specific target.
 */
function parseHelmYaml(src: string): Record<string, string> {
  const doc = yaml.load(src) as Record<string, unknown> | undefined;
  const featureFlags = (doc?.platformConfig as Record<string, unknown>)?.studio as
    | Record<string, unknown>
    | undefined;
  const flags = featureFlags?.feature_flags as Record<string, unknown> | undefined;
  if (!flags) return {};
  return Object.fromEntries(
    Object.entries(flags).map(([k, v]) => [snakeToCamel(k), String(v).trim()])
  );
}

// ---------------------------------------------------------------------------
// Read all sources
// ---------------------------------------------------------------------------

function safeRead<T>(filepath: string, parser: (src: string) => T, fallback: T): T {
  try {
    return parser(readFile(filepath));
  } catch (err) {
    console.error(`Failed to read ${path.relative(REPO_ROOT, filepath)}: ${err}`);
    return fallback;
  }
}

const studioDefaults = safeRead(FEATURE_FLAGS_TS, parseStudioDefaults, {});
const localEnvOverrides = fs.existsSync(LOCAL_ENV_FILE)
  ? safeRead(LOCAL_ENV_FILE, parseLocalEnv, {})
  : {};
const envMappingDefaults = safeRead(ENV_MAPPINGS_PY, parseEnvMappings, {});

const ciFiles =
  HELM_DEPLOYMENT_VALUES_DIR && fs.existsSync(HELM_DEPLOYMENT_VALUES_DIR)
    ? fs
        .readdirSync(HELM_DEPLOYMENT_VALUES_DIR)
        .filter((f) => f.endsWith('.yaml'))
        .sort()
    : [];
const ciOverrides: Record<string, Record<string, string>> = {};
for (const file of ciFiles) {
  ciOverrides[file] = safeRead(path.join(HELM_DEPLOYMENT_VALUES_DIR, file), parseHelmYaml, {});
}

const allFlags = [
  ...new Set([
    ...Object.keys(studioDefaults),
    ...Object.keys(localEnvOverrides),
    ...Object.keys(envMappingDefaults),
  ]),
].sort();

// ---------------------------------------------------------------------------
// Resolve effective values
// ---------------------------------------------------------------------------

function resolveLocal(flag: string): string {
  return localEnvOverrides[flag] ?? studioDefaults[flag] ?? 'false';
}

function resolveDeployment(flag: string, yamlFile: string): string {
  return ciOverrides[yamlFile]?.[flag] ?? envMappingDefaults[flag] ?? 'false';
}

// ---------------------------------------------------------------------------
// Repo-relative paths (for JSON output)
// ---------------------------------------------------------------------------

const rel = (abs: string) => path.relative(REPO_ROOT, abs);
const REL_FEATURE_FLAGS_TS = rel(FEATURE_FLAGS_TS);
const REL_LOCAL_ENV_FILE = rel(LOCAL_ENV_FILE);
const REL_ENV_MAPPINGS_PY = rel(ENV_MAPPINGS_PY);

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

if (process.argv.includes('--json')) {
  // Machine-readable JSON for agents
  type Source = { path: string; value: string | null };
  type FlagEntry = {
    local: { value: string; override: Source; default: Source };
    deployments: Record<string, { value: string; override: Source } | Source>;
  };

  const output: Record<string, FlagEntry> = {};
  for (const flag of allFlags) {
    const deployments: FlagEntry['deployments'] = {};
    for (const file of ciFiles) {
      deployments[envName(file)] = {
        value: resolveDeployment(flag, file),
        override: {
          path: rel(path.join(HELM_DEPLOYMENT_VALUES_DIR, file)),
          value: ciOverrides[file]?.[flag] ?? null,
        },
      };
    }
    deployments['default'] = {
      path: REL_ENV_MAPPINGS_PY,
      value: envMappingDefaults[flag] ?? null,
    };

    output[flag] = {
      local: {
        value: resolveLocal(flag),
        override: { path: REL_LOCAL_ENV_FILE, value: localEnvOverrides[flag] ?? null },
        default: { path: REL_FEATURE_FLAGS_TS, value: studioDefaults[flag] ?? null },
      },
      deployments,
    };
  }

  console.log(JSON.stringify(output, null, 2));
} else {
  // CLI color-coded per-flag cards
  const UNSET = '·';
  const W = Math.max(28, ...ciFiles.map((f) => f.length + 6));

  const colorize = (v: string): string => {
    if (v === UNSET) return `\x1b[2m${v}\x1b[0m`;
    if (v === 'false') return `\x1b[31m${v}\x1b[0m`;
    if (v === 'preview') return `\x1b[33m${v}\x1b[0m`;
    return `\x1b[32m${v}\x1b[0m`;
  };
  const pad = (s: string, w: number) => s + ' '.repeat(Math.max(0, w - s.length));
  const row = (label: string, v: string, indent = 2) =>
    console.log(' '.repeat(indent) + pad(label, W - indent) + v);
  const val = (v: string | undefined) => colorize(v !== undefined ? v : UNSET);

  for (const flag of allFlags) {
    console.log(`\x1b[1m═══ ${flag} ${'═'.repeat(Math.max(0, 48 - flag.length))}\x1b[0m`);
    console.log('');

    // Local
    row('LOCAL', val(resolveLocal(flag)));
    row('.env.dev.local', val(localEnvOverrides[flag]), 4);
    console.log('  (default)');
    row('featureFlags.ts', val(studioDefaults[flag]), 4);
    console.log('');

    // Deployment targets
    for (const file of ciFiles) {
      row(envName(file), val(resolveDeployment(flag, file)));
      row(file, val(ciOverrides[file]?.[flag]), 4);
      console.log('');
    }

    console.log('  (default)');
    row('env_mappings.py', val(envMappingDefaults[flag]), 4);
    console.log('');
  }
}
