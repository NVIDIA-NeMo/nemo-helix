// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { generateDataDesignerJobRequestTool } from '@studio/components/NewDataDesignerJobForm/tools';

interface ObjectSchema {
  required?: string[];
  properties?: Record<string, ObjectSchema>;
  additionalProperties?: boolean | object;
  items?: ObjectSchema;
  anyOf?: ObjectSchema[];
  enum?: string[];
}

const emitted: ObjectSchema = JSON.parse(
  JSON.stringify(generateDataDesignerJobRequestTool.function.parameters)
);

const childAt = (schema: ObjectSchema, ...path: string[]): ObjectSchema =>
  path.reduce<ObjectSchema>((node, key) => {
    const next = node.properties?.[key];
    if (!next) {
      throw new Error(`No property "${key}" in emitted tool schema`);
    }
    return next;
  }, schema);

const keysOf = (schema: ObjectSchema): string[] => Object.keys(schema.properties ?? {});

const jobRequest = childAt(emitted, 'job_request');
const spec = childAt(jobRequest, 'spec');
const config = childAt(spec, 'config');
const columnSchemas = childAt(config, 'columns').items?.anyOf ?? [];

const columnSchemaFor = (columnType: string): ObjectSchema => {
  const schema = columnSchemas.find(
    (column) => column.properties?.column_type?.enum?.[0] === columnType
  );
  if (!schema) {
    throw new Error(`No column schema for column_type "${columnType}"`);
  }
  return schema;
};

const objectSchemasIn = (schema: ObjectSchema): ObjectSchema[] => [
  ...(schema.properties ? [schema] : []),
  ...Object.values(schema.properties ?? {}).flatMap(objectSchemasIn),
  ...(schema.items ? objectSchemasIn(schema.items) : []),
  ...(schema.anyOf ?? []).flatMap(objectSchemasIn),
];

describe('generateDataDesignerJobRequestTool schema order', () => {
  it('orders the job request, spec and config keys', () => {
    expect(keysOf(jobRequest)).toEqual(['name', 'spec']);
    expect(keysOf(spec)).toEqual(['num_records', 'config']);
    expect(keysOf(config)).toEqual(['model_configs', 'columns', 'seed_config']);
    expect(config.required).toEqual(['model_configs', 'columns']);
  });

  it.each([
    ['sampler', ['name', 'column_type', 'sampler_type', 'params', 'convert_to']],
    ['expression', ['name', 'column_type', 'expr', 'dtype']],
    [
      'llm-text',
      [
        'name',
        'column_type',
        'prompt',
        'model_alias',
        'system_prompt',
        'extract_reasoning_content',
      ],
    ],
    [
      'llm-structured',
      [
        'name',
        'column_type',
        'prompt',
        'model_alias',
        'output_format',
        'system_prompt',
        'extract_reasoning_content',
      ],
    ],
    [
      'llm-judge',
      [
        'name',
        'column_type',
        'prompt',
        'model_alias',
        'scores',
        'system_prompt',
        'extract_reasoning_content',
      ],
    ],
    [
      'llm-code',
      [
        'name',
        'column_type',
        'prompt',
        'model_alias',
        'code_lang',
        'system_prompt',
        'extract_reasoning_content',
      ],
    ],
    ['validation', ['name', 'column_type', 'target_columns', 'validator_type', 'validator_params']],
  ])('orders %s column keys and rejects unknown ones', (columnType, expectedKeys) => {
    const column = columnSchemaFor(columnType);
    expect(keysOf(column)).toEqual(expectedKeys);
    expect(column.additionalProperties).toBe(false);
  });

  it('offers exactly one schema per column type', () => {
    expect(columnSchemas.map((column) => column.properties?.column_type?.enum)).toEqual([
      ['sampler'],
      ['expression'],
      ['llm-text'],
      ['llm-structured'],
      ['llm-judge'],
      ['llm-code'],
      ['validation'],
    ]);
  });

  it('only offers seed sources the builder can load', () => {
    const source = childAt(config, 'seed_config', 'source');
    expect(keysOf(source)).toEqual(['seed_type', 'path']);
    expect(childAt(source, 'seed_type').enum).toEqual(['nhx']);
  });

  it('declares required keys before optional ones in every object', () => {
    for (const schema of objectSchemasIn(emitted)) {
      const required = schema.required ?? [];
      expect(keysOf(schema).slice(0, required.length)).toEqual(required);
    }
  });
});
