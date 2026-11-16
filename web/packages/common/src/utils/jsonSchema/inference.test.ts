// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { INFERENCE_MAX_DEPTH, inferJsonSchema } from '@nemo/common/src/utils/jsonSchema/inference';

const SCHEMA = 'https://json-schema.org/draft/2020-12/schema';

describe('inferJsonSchema', () => {
  it('stamps $schema on every output', () => {
    expect(inferJsonSchema({ a: 1 })).toMatchObject({ $schema: SCHEMA });
    expect(inferJsonSchema([1, 2, 3])).toMatchObject({ $schema: SCHEMA });
    expect(inferJsonSchema('hello')).toMatchObject({ $schema: SCHEMA });
  });

  it('infers primitive types', () => {
    expect(inferJsonSchema('hi')).toMatchObject({ type: 'string' });
    expect(inferJsonSchema(true)).toMatchObject({ type: 'boolean' });
    expect(inferJsonSchema(42)).toMatchObject({ type: 'integer' });
    expect(inferJsonSchema(3.14)).toMatchObject({ type: 'number' });
  });

  it('emits {} for null (no constraint) instead of type: "null"', () => {
    const schema = inferJsonSchema(null);
    expect(schema).toEqual({ $schema: SCHEMA });
    expect(schema.type).toBeUndefined();
  });

  it('omits the required array on inferred object schemas', () => {
    const schema = inferJsonSchema({ name: 'a', age: 1 });
    expect(schema.required).toBeUndefined();
    expect(schema).toMatchObject({
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'integer' },
      },
    });
  });

  it('recursively infers nested object properties', () => {
    const schema = inferJsonSchema({
      user: { id: 1, profile: { name: 'a', active: true } },
    });
    expect(schema).toMatchObject({
      type: 'object',
      properties: {
        user: {
          type: 'object',
          properties: {
            id: { type: 'integer' },
            profile: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                active: { type: 'boolean' },
              },
            },
          },
        },
      },
    });
  });

  it('caps recursion at maxDepth, replacing deeper objects with bare type: "object"', () => {
    // Build an object 3 levels deep, then cap at 2.
    const deep = { a: { b: { c: 1 } } };
    const schema = inferJsonSchema(deep, { maxDepth: 2 });
    expect(schema).toMatchObject({
      type: 'object',
      properties: {
        a: {
          type: 'object',
          properties: {
            b: { type: 'object' },
          },
        },
      },
    });
    // The cap stripped further descent: `b` has no `properties` key.
    const bSchema = (schema.properties as Record<string, Record<string, unknown>>).a.properties;
    expect((bSchema as Record<string, Record<string, unknown>>).b.properties).toBeUndefined();
  });

  it('exposes INFERENCE_MAX_DEPTH as a positive constant', () => {
    expect(INFERENCE_MAX_DEPTH).toBeGreaterThan(0);
  });

  describe('arrays', () => {
    it('infers items.type when every element is the same primitive', () => {
      expect(inferJsonSchema([1, 2, 3])).toMatchObject({
        type: 'array',
        items: { type: 'integer' },
      });
      expect(inferJsonSchema(['a', 'b'])).toMatchObject({
        type: 'array',
        items: { type: 'string' },
      });
    });

    it('emits items: {} for empty arrays', () => {
      expect(inferJsonSchema([])).toMatchObject({ type: 'array', items: {} });
    });

    it('emits items: {} for mixed-type arrays', () => {
      expect(inferJsonSchema([1, 'two'])).toMatchObject({ type: 'array', items: {} });
    });

    it('emits items: {} when an integer and a float coexist (no implicit promotion)', () => {
      expect(inferJsonSchema([1, 2.5])).toMatchObject({ type: 'array', items: {} });
    });

    it('infers items as the first-element shape when the array is homogeneously objects', () => {
      expect(inferJsonSchema([{ a: 1 }, { a: 2 }])).toMatchObject({
        type: 'array',
        items: {
          type: 'object',
          properties: { a: { type: 'integer' } },
        },
      });
    });

    it('emits items: {} when the array mixes objects with non-objects', () => {
      expect(inferJsonSchema([{ a: 1 }, 'two'])).toMatchObject({
        type: 'array',
        items: {},
      });
    });
  });
});
