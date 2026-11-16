// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';
import { collectSchemaExtension, SCHEMA_EXTENSIONS } from './generate-schema-extensions';

const KEY = 'x-learning-rate-defaults';
const extension = SCHEMA_EXTENSIONS[KEY];
const rules = [{ when: { 'training.finetuning_type': 'lora' }, value: 1e-4 }];

const specWith = (schemas: Record<string, unknown>) => ({ components: { schemas } });

describe('collectSchemaExtension', () => {
  it('collects the schemas that carry the key and ignores the rest', () => {
    const spec = specWith({ Tagged: { type: 'object', [KEY]: rules }, Plain: { type: 'object' } });

    expect(collectSchemaExtension(spec, KEY, extension)).toEqual({ Tagged: rules });
  });

  it('returns nothing for a spec without component schemas', () => {
    expect(collectSchemaExtension({ paths: {} }, KEY, extension)).toEqual({});
  });

  it('throws on a malformed rule', () => {
    const spec = specWith({ Tagged: { [KEY]: [{ when: {}, value: 'fast' }] } });

    expect(() => collectSchemaExtension(spec, KEY, extension)).toThrow(
      `Tagged.${KEY} must be a list of {when, value} rules`
    );
  });
});

describe('x-learning-rate-defaults validator', () => {
  it('rejects a non-string `when` value', () => {
    expect(extension.validate([{ when: { 'lora.rank': 8 }, value: 1e-4 }])).toBe(false);
  });

  it('accepts well-formed rules', () => {
    expect(extension.validate(rules)).toBe(true);
  });
});
