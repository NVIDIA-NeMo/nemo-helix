// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * A feature flag is only controllable end-to-end if it is declared in all three
 * registries: the Studio flag definitions, the `.env.fastapi` marker file that the
 * Vite build bakes into the bundle, and the `env_mappings.py` table that substitutes
 * those markers at serve time.
 *
 * Missing the `.env.fastapi` entry is silent — the flag still compiles, still has an
 * `env_mappings.py` row, and still reads as its code default. The only symptom is that
 * `studio.feature_flags.<key>` stops doing anything in a deployed Studio. These tests
 * turn that into a failure at the point the flag is added.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const THIS_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(THIS_DIR, '../../../../../..');

const read = (relativePath: string) =>
  fs.readFileSync(path.join(REPO_ROOT, relativePath), 'utf-8');

const sorted = (values: Iterable<string>) => [...new Set(values)].sort();

const matches = (source: string, pattern: RegExp) =>
  sorted([...source.matchAll(pattern)].map((match) => match[1]));

/** Env vars named inside the `flagDefinitions` object, ignoring the how-to comments above it. */
const flagDefinitionEnvVars = (): string[] => {
  const source = read('web/packages/studio/src/constants/featureFlags/featureFlags.ts');
  const block = /export const flagDefinitions = \{([\s\S]*?)\n\} as const;/.exec(source);
  if (!block) throw new Error('Could not locate the flagDefinitions object in featureFlags.ts');
  return matches(block[1], /'(VITE_FF_[A-Z0-9_]+)'/g);
};

const fastApiMarkerEnvVars = (): string[] =>
  matches(read('web/packages/studio/env/.env.fastapi'), /^(VITE_FF_[A-Z0-9_]+)=/gm);

const envMappingEnvVars = (): string[] =>
  matches(
    read('services/studio/src/nhx/studio/env_mappings.py'),
    /marker="STUDIO_UI_(VITE_FF_[A-Z0-9_]+)"/g
  );

describe('feature flag registry parity', () => {
  it('declares every flag in .env.fastapi so the build emits a substitutable marker', () => {
    expect(fastApiMarkerEnvVars()).toEqual(flagDefinitionEnvVars());
  });

  it('declares every flag in env_mappings.py so the marker resolves from platform config', () => {
    expect(envMappingEnvVars()).toEqual(flagDefinitionEnvVars());
  });
});
