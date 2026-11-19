// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Emits the `x-*` schema extensions Orval drops to `generated/schema-extensions/`.
// Run by `gen:<service>` for every service that lists `schemaExtensions`.
import fs from 'fs';
import path from 'path';
import { parse as parseYaml } from 'yaml';
import { serviceConfigs } from './constants';
import { isRecord } from './utils';

interface SchemaExtension {
  readonly module: string;
  readonly exportName: string;
  /** TS declarations emitted above the export. */
  readonly typeDecl: readonly string[];
  readonly valueType: string;
  /** Shape named in the error when `validate` fails. */
  readonly expected: string;
  readonly validate: (value: unknown) => boolean;
}

const isLearningRateDefaultRule = (rule: unknown): boolean =>
  isRecord(rule) &&
  typeof rule.value === 'number' &&
  isRecord(rule.when) &&
  Object.values(rule.when).every((v) => typeof v === 'string');

export const SCHEMA_EXTENSIONS: Readonly<Record<string, SchemaExtension>> = {
  'x-learning-rate-defaults': {
    module: 'learningRateDefaults',
    exportName: 'LEARNING_RATE_DEFAULTS',
    typeDecl: [
      '/** The rate applies when every dotted path in `when`, relative to the schema, equals its value. */',
      'export interface LearningRateDefaultRule {',
      '  readonly when: Readonly<Record<string, string>>;',
      '  readonly value: number;',
      '}',
      '',
      '/** Ordered rules per schema; the first match wins, and no match means the field default. */',
    ],
    valueType: 'readonly LearningRateDefaultRule[]',
    expected: 'a list of {when, value} rules',
    validate: (value) => Array.isArray(value) && value.every(isLearningRateDefaultRule),
  },
};

/** Collect `key` from each component schema that has it, keyed by schema name. */
export function collectSchemaExtension(
  spec: unknown,
  key: string,
  extension: Pick<SchemaExtension, 'expected' | 'validate'>
): Record<string, unknown> {
  const schemas = isRecord(spec) && isRecord(spec.components) ? spec.components.schemas : undefined;
  if (!isRecord(schemas)) {
    return {};
  }

  const collected: Record<string, unknown> = {};
  for (const [name, schema] of Object.entries(schemas)) {
    const value = isRecord(schema) ? schema[key] : undefined;
    if (value === undefined) {
      continue;
    }
    if (!extension.validate(value)) {
      throw new Error(`${name}.${key} must be ${extension.expected}`);
    }
    collected[name] = value;
  }
  return collected;
}

const renderModule = (key: string, extension: SchemaExtension, values: Record<string, unknown>) =>
  [
    '/**',
    ' * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.',
    ' * SPDX-License-Identifier: Apache-2.0',
    ' *',
    ` * Auto-generated from the \`${key}\` OpenAPI extension.`,
    ' * Do not edit manually.',
    ' */',
    '',
    ...extension.typeDecl,
    `export const ${extension.exportName} = ${JSON.stringify(values, null, 2)} as const satisfies Readonly<`,
    `  Record<string, ${extension.valueType}>`,
    '>;',
    '',
  ].join('\n');

const main = () => {
  const requested = process.argv.slice(2);
  const services = requested.length > 0 ? requested : Object.keys(serviceConfigs);

  const byKey: Record<string, Record<string, unknown>> = {};
  for (const service of services) {
    const config = serviceConfigs[service];
    if (!config) {
      throw new Error(
        `Unknown service "${service}". Known: ${Object.keys(serviceConfigs).join(', ')}`
      );
    }
    if (!config.schemaExtensions?.length) {
      continue;
    }

    const spec: unknown = parseYaml(fs.readFileSync(path.resolve(__dirname, config.url), 'utf8'));
    for (const key of config.schemaExtensions) {
      const extension = SCHEMA_EXTENSIONS[key];
      if (!extension) {
        throw new Error(`${service} opts into unknown schema extension "${key}"`);
      }
      const merged = (byKey[key] ??= {});
      for (const [name, value] of Object.entries(collectSchemaExtension(spec, key, extension))) {
        if (name in merged) {
          throw new Error(`${key}: schema "${name}" appears in more than one service`);
        }
        merged[name] = value;
      }
    }
  }

  const outDir = path.resolve(__dirname, '../generated/schema-extensions');
  fs.mkdirSync(outDir, { recursive: true });
  for (const [key, values] of Object.entries(byKey)) {
    if (Object.keys(values).length === 0) {
      continue;
    }
    const extension = SCHEMA_EXTENSIONS[key];
    fs.writeFileSync(
      path.join(outDir, `${extension.module}.ts`),
      renderModule(key, extension, values)
    );
    console.log(`✓ Wrote ${key} for ${Object.keys(values).length} schema(s)`);
  }
};

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  main();
}
