#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable no-console -- CLI script */
/**
 * Shifts the step timestamps in the sample ATIF traces forward.
 *
 * Intake rejects a span batch whose `started_at` falls outside the ClickHouse TTL (90 days by
 * default), so these fixed-date sample traces expire. This rolls the whole set forward while
 * preserving the spacing between traces and between steps within a trace.
 *
 *   pnpm traces:bump                    # newest step -> ~1 day ago
 *   pnpm traces:bump --days 30          # shift everything forward 30 days
 *   pnpm traces:bump --dry-run          # print the shift, write nothing
 */
import type { AtifIngestRequest } from '@nemo/sdk/generated/platform/schema';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TRACES_DIR = resolve(__dirname, '../src/mocks/email-security-triage-traces');

const MS_PER_DAY = 24 * 60 * 60 * 1000;

interface LoadedTrace {
  path: string;
  trace: AtifIngestRequest;
}

/** `2026-08-31T21:50:23Z` — the form the sample traces are written in. */
const toIso = (moment: Date): string => `${moment.toISOString().slice(0, 19)}Z`;

const stepTimestamps = (trace: AtifIngestRequest): string[] =>
  (trace.steps ?? []).flatMap((step) => (step.timestamp ? [step.timestamp] : []));

const loadTraces = async (paths: string[]): Promise<LoadedTrace[]> =>
  Promise.all(
    paths.map(async (path) => ({
      path,
      trace: JSON.parse(await readFile(path, 'utf8')) as AtifIngestRequest,
    }))
  );

const defaultPaths = async (): Promise<string[]> => {
  const names = await readdir(TRACES_DIR);
  return names
    .filter((name) => name.startsWith('trace-') && name.endsWith('.json'))
    .sort()
    .map((name) => join(TRACES_DIR, name));
};

const main = async () => {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      days: { type: 'string' },
      'newest-days-ago': { type: 'string', default: '1' },
      'dry-run': { type: 'boolean', default: false },
    },
  });

  if (values.days !== undefined && values['newest-days-ago'] !== '1') {
    throw new Error('Pass either --days or --newest-days-ago, not both.');
  }

  const paths =
    positionals.length > 0 ? positionals.map((path) => resolve(path)) : await defaultPaths();
  if (paths.length === 0) throw new Error(`No trace files found in ${TRACES_DIR}`);

  const traces = await loadTraces(paths);
  const stamps = traces.flatMap(({ trace }) =>
    stepTimestamps(trace).map((stamp) => Date.parse(stamp))
  );
  if (stamps.length === 0) throw new Error('No step timestamps found.');

  let shiftMs: number;
  if (values.days !== undefined) {
    shiftMs = Number(values.days) * MS_PER_DAY;
    if (!Number.isFinite(shiftMs)) throw new Error(`--days must be a number, got "${values.days}"`);
  } else {
    const newestDaysAgo = Number(values['newest-days-ago']);
    if (!Number.isFinite(newestDaysAgo)) {
      throw new Error(`--newest-days-ago must be a number, got "${values['newest-days-ago']}"`);
    }
    shiftMs = Date.now() - newestDaysAgo * MS_PER_DAY - Math.max(...stamps);
    if (shiftMs <= 0) {
      throw new Error(
        `Traces are already newer than the target; this would shift backwards by ${-shiftMs}ms. ` +
          'Pass an explicit --days to do it anyway.'
      );
    }
  }

  console.log(`shifting ${paths.length} file(s) by ${(shiftMs / MS_PER_DAY).toFixed(4)} day(s)`);

  for (const { path, trace } of traces) {
    for (const step of trace.steps ?? []) {
      if (step.timestamp) step.timestamp = toIso(new Date(Date.parse(step.timestamp) + shiftMs));
    }
    console.log(`  ${basename(path)}: ${stepTimestamps(trace).join(' ')}`);
    if (!values['dry-run']) await writeFile(path, `${JSON.stringify(trace, null, 2)}\n`);
  }

  if (values['dry-run']) console.log('dry run: nothing written');
};

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
