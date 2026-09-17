#!/usr/bin/env tsx
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Generates types for all OpenAPI specs in the NeMo Platform repository.
 *
 * The generated tree under `./generated/` is gitignored. To keep `pnpm install`
 * cheap, this script writes a content-hash sentinel after a successful run and
 * skips regeneration when the inputs haven't changed. Pass `--force` to bypass.
 */

import crypto from 'crypto';
import { execSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { serviceConfigs } from './orval/constants';

const FORCE = process.argv.includes('--force');
const GENERATED_DIR = path.join(__dirname, 'generated');
const HASH_FILE = path.join(GENERATED_DIR, '.input-hash');
const ORVAL_DIR = path.join(__dirname, 'orval');
const MIN_SERVICE_PIPELINES = 4;
const MAX_PROCESSES_ENV = 'WEB_SDK_GEN_MAX_PROCESSES';
const MAX_PROCESSES_PATTERN = /^(\d+)(%)?$/;

const services = Object.keys(serviceConfigs) as Array<keyof typeof serviceConfigs>;

const availableParallelism = (): number =>
  typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length;

const defaultMaxProcesses = (): number => {
  return Math.min(services.length, Math.max(MIN_SERVICE_PIPELINES, availableParallelism()));
};

const maxProcesses = (): string => {
  const configured = process.env[MAX_PROCESSES_ENV]?.trim();
  if (configured) {
    const match = configured.match(MAX_PROCESSES_PATTERN);
    const value = match ? Number(match[1]) : Number.NaN;
    if (!match || value < 1) {
      throw new Error(
        `${MAX_PROCESSES_ENV} must be a positive count or percent, got "${configured}"`
      );
    }

    const configuredProcesses = match[2]
      ? Math.round((availableParallelism() * value) / 100)
      : value;
    return String(Math.min(services.length, Math.max(1, configuredProcesses)));
  }
  return String(defaultMaxProcesses());
};

interface GenerationConfig {
  service: string;
  zod?: boolean;
}

const generationConfigs: GenerationConfig[] = [
  ...services.map((service) => ({ service, zod: serviceConfigs[service]?.zod })),
];

const generateCommands = (config: GenerationConfig) => {
  const { service, zod } = config;

  const baseCommand = `pnpm run gen:${service}`;
  if (zod) {
    return `${baseCommand} && pnpm run gen:${service}-zod`;
  }
  return baseCommand;
};

const readOrvalVersion = (): string => {
  try {
    const orvalPkg = JSON.parse(
      fs.readFileSync(path.join(__dirname, 'node_modules', 'orval', 'package.json'), 'utf8')
    );
    return String(orvalPkg.version ?? 'unknown');
  } catch {
    return 'unknown';
  }
};

/**
 * Hash inputs to generation: all spec YAMLs, the orval version, and the
 * generator source files. Any of these changing invalidates the cache.
 */
const computeInputHash = (): string => {
  const hash = crypto.createHash('sha256');

  for (const [service, config] of Object.entries(serviceConfigs)) {
    hash.update(`service:${service}\n`);
    if (config.url.startsWith('http')) {
      // Remote specs aren't cacheable here; including the URL still lets the
      // hash invalidate when the URL itself changes.
      hash.update(`url:${config.url}\n`);
      continue;
    }
    const filePath = path.resolve(ORVAL_DIR, config.url);
    hash.update(fs.readFileSync(filePath));
    hash.update('\n');
  }

  hash.update(`orval:${readOrvalVersion()}\n`);

  const generatorSources = [
    path.join(__dirname, 'orval.config.ts'),
    path.join(ORVAL_DIR, 'generate.ts'),
    path.join(ORVAL_DIR, 'constants.ts'),
    path.join(ORVAL_DIR, 'format-generated.ts'),
    path.join(ORVAL_DIR, 'generateCustomFetcher.ts'),
    path.join(ORVAL_DIR, 'operationNameOverride.ts'),
    path.join(ORVAL_DIR, 'generate-capabilities.ts'),
    path.join(__dirname, 'generateAll.ts'),
  ];
  for (const file of generatorSources) {
    if (fs.existsSync(file)) {
      hash.update(fs.readFileSync(file));
      hash.update('\n');
    }
  }

  return hash.digest('hex');
};

/**
 * The cache is valid only when (1) the hash file exists and matches the
 * current input hash, and (2) the generated tree actually has content beyond
 * the sentinel — protects against partially-deleted output.
 */
const isCacheValid = (currentHash: string): boolean => {
  if (!fs.existsSync(HASH_FILE)) return false;
  if (!fs.existsSync(GENERATED_DIR)) return false;
  const entries = fs.readdirSync(GENERATED_DIR).filter((name) => name !== '.input-hash');
  if (entries.length === 0) return false;
  const stored = fs.readFileSync(HASH_FILE, 'utf8').trim();
  return stored === currentHash;
};

const writeHash = (hash: string) => {
  fs.mkdirSync(GENERATED_DIR, { recursive: true });
  fs.writeFileSync(HASH_FILE, hash);
};

const main = async () => {
  const currentHash = computeInputHash();

  if (!FORCE && isCacheValid(currentHash)) {
    console.log('✓ SDK is up to date (input hash matches). Skipping generation.');
    console.log('  Pass --force to regenerate anyway.');
    return;
  }

  if (FORCE) {
    console.log('🔁 --force passed; regenerating regardless of input hash.\n');
  } else {
    console.log('🚀 Inputs changed (or no cached hash). Regenerating SDK...\n');
  }

  const commands = generationConfigs.map(generateCommands);
  const serviceNames = generationConfigs.map((config) => config.service);
  const maxServicePipelines = maxProcesses();
  const colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'purple', 'white', 'gray'];

  const concurrentlyArgs = [
    '--max-processes',
    maxServicePipelines,
    '--names',
    serviceNames.join(','),
    '-c',
    colors.slice(0, serviceNames.length).join(','),
    ...commands.map((cmd) => `"${cmd}"`),
  ];

  const concurrentlyCommand = `pnpm exec concurrently ${concurrentlyArgs.join(' ')}`;

  console.log(`Executing commands in parallel (max ${maxServicePipelines} service pipelines):`);
  commands.forEach((cmd, index) => {
    console.log(`  ${index + 1}. ${cmd}`);
  });
  console.log('');

  try {
    execSync(concurrentlyCommand, { stdio: 'inherit' });
    console.log('\n🧰 Generating LLM capability registry...');
    execSync('pnpm run gen:capabilities', { stdio: 'inherit' });
    writeHash(currentHash);
    console.log('\n🎉 All type generation completed successfully!');
  } catch {
    console.error('\n💥 Some type generation failed. Check the output above for details.');
    process.exit(1);
  }
};

main().catch((error) => {
  console.error('💥 Fatal error during type generation:', error);
  process.exit(1);
});
