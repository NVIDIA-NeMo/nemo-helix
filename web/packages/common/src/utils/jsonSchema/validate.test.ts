// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  parseAndValidate,
  validateJsonSchemaDocument,
} from '@nemo/common/src/utils/jsonSchema/validate';

describe('validateJsonSchemaDocument', () => {
  it('accepts a minimal valid Draft 2020-12 schema', () => {
    const result = validateJsonSchemaDocument({
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      type: 'object',
      properties: { name: { type: 'string' } },
    });
    expect(result.valid).toBe(true);
    expect(result.errors).toEqual([]);
  });

  it('accepts a bare {} (no constraints)', () => {
    expect(validateJsonSchemaDocument({}).valid).toBe(true);
  });

  it('rejects an unknown primitive type', () => {
    const result = validateJsonSchemaDocument({ type: 'foobar' });
    expect(result.valid).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  it('rejects a non-object schema document', () => {
    expect(validateJsonSchemaDocument('not a schema').valid).toBe(false);
    expect(validateJsonSchemaDocument(42).valid).toBe(false);
  });
});

describe('parseAndValidate', () => {
  it('returns the parsed object when the text is valid JSON + valid schema', () => {
    const result = parseAndValidate('{"type": "string"}');
    expect(result.valid).toBe(true);
    if (result.valid) {
      expect(result.value).toEqual({ type: 'string' });
    }
  });

  it('reports parse errors when the text is not valid JSON', () => {
    const result = parseAndValidate('{not json}');
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors[0]).toMatch(/Invalid JSON/);
    }
  });

  it('rejects valid JSON that is not a JSON object (e.g. arrays, primitives)', () => {
    const arr = parseAndValidate('[1, 2, 3]');
    expect(arr.valid).toBe(false);
    const prim = parseAndValidate('42');
    expect(prim.valid).toBe(false);
  });

  it('forwards schema-validity errors from validateJsonSchemaDocument', () => {
    const result = parseAndValidate('{"type": "foobar"}');
    expect(result.valid).toBe(false);
  });
});
