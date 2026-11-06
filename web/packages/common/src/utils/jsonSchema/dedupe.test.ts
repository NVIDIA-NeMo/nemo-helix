// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { buildDatasetMetadata } from '@nemo/common/src/utils/jsonSchema/dedupe';

const S1 = { type: 'object', properties: { a: { type: 'integer' } } };
const S2 = { type: 'object', properties: { b: { type: 'string' } } };
const S3 = { type: 'object', properties: { c: { type: 'boolean' } } };

describe('buildDatasetMetadata', () => {
  it('returns {} for empty perFile and no existing', () => {
    expect(buildDatasetMetadata([])).toEqual({});
  });

  it('deep-clones existing when perFile is empty (no shared nested references)', () => {
    const existing = {
      schema: 'schema_1' as const,
      schema_defs: { schema_1: S1 },
      schemas_by_path: { 'a.jsonl': 'schema_1' as const },
    };
    const result = buildDatasetMetadata([], existing);
    expect(result).toEqual(existing);
    // Outer containers and nested schema objects are independent copies.
    expect(result.schema_defs).not.toBe(existing.schema_defs);
    expect(result.schemas_by_path).not.toBe(existing.schemas_by_path);
    expect(result.schema_defs?.schema_1).not.toBe(existing.schema_defs.schema_1);
    // Mutating the clone does not bleed into the original.
    (result.schema_defs!.schema_1 as Record<string, unknown>).mutated = true;
    expect(existing.schema_defs.schema_1).toEqual(S1);
  });

  it('emits inline schema for a single new file with no existing', () => {
    const result = buildDatasetMetadata([{ path: 'a.jsonl', schema: S1 }]);
    expect(result).toEqual({ schema: S1, schema_defs: {}, schemas_by_path: {} });
  });

  it('emits inline schema for N identical new files with no existing', () => {
    const result = buildDatasetMetadata([
      { path: 'a.jsonl', schema: S1 },
      { path: 'b.jsonl', schema: S1 },
      { path: 'c.jsonl', schema: S1 },
    ]);
    expect(result).toEqual({ schema: S1, schema_defs: {}, schemas_by_path: {} });
  });

  it('considers structural equality even if object key orders differ', () => {
    const reordered = { properties: { a: { type: 'integer' } }, type: 'object' };
    const result = buildDatasetMetadata([
      { path: 'a.jsonl', schema: S1 },
      { path: 'b.jsonl', schema: reordered },
    ]);
    expect(result.schema_defs).toEqual({});
    expect(result.schemas_by_path).toEqual({});
  });

  it('dedupes 2 distinct schemas into schema_defs with top-level ref to dominant', () => {
    // S2 appears twice -> dominant.
    const result = buildDatasetMetadata([
      { path: 'a.jsonl', schema: S1 },
      { path: 'b.jsonl', schema: S2 },
      { path: 'c.jsonl', schema: S2 },
    ]);
    expect(result.schema_defs).toMatchObject({
      schema_1: S1,
      schema_2: S2,
    });
    expect(result.schema).toBe('schema_2');
    // Only the divergent path is listed.
    expect(result.schemas_by_path).toEqual({ 'a.jsonl': 'schema_1' });
  });

  it('dedupes 3-way split (no dominant) with first-encountered winning the top-level tie', () => {
    const result = buildDatasetMetadata([
      { path: 'a.jsonl', schema: S1 },
      { path: 'b.jsonl', schema: S2 },
      { path: 'c.jsonl', schema: S3 },
    ]);
    expect(result.schema).toBe('schema_1');
    expect(result.schema_defs).toMatchObject({
      schema_1: S1,
      schema_2: S2,
      schema_3: S3,
    });
    expect(result.schemas_by_path).toEqual({
      'b.jsonl': 'schema_2',
      'c.jsonl': 'schema_3',
    });
  });

  it('merges with existing: reuses matching def key for one match, mints new for novelty', () => {
    const existing = {
      schema: 'schema_1' as const,
      schema_defs: { schema_1: S1 },
      schemas_by_path: { 'old.jsonl': 'schema_1' as const },
    };
    const result = buildDatasetMetadata(
      [
        { path: 'new.jsonl', schema: S1 },
        { path: 'other.jsonl', schema: S2 },
      ],
      existing
    );
    expect(result.schema_defs).toMatchObject({
      schema_1: S1,
      schema_2: S2,
    });
    // Existing top-level (schema_1) preserved as the default.
    expect(result.schema).toBe('schema_1');
    // Only the divergent new file is listed; matching files inherit top-level.
    expect(result.schemas_by_path).toEqual({ 'other.jsonl': 'schema_2' });
  });

  it('mints the lowest unused integer when minting new def keys (fills gaps)', () => {
    const existing = {
      schema_defs: { schema_5: S1 },
      schemas_by_path: {},
    };
    const result = buildDatasetMetadata([{ path: 'a.jsonl', schema: S2 }], existing);
    // Existing schema_5 preserved. New mint fills the lowest gap (schema_1),
    // keeping the schema_N namespace compact.
    expect(Object.keys(result.schema_defs ?? {}).sort()).toEqual(['schema_1', 'schema_5']);
    expect(result.schema_defs?.schema_5).toEqual(S1);
    expect(result.schema_defs?.schema_1).toEqual(S2);
  });

  it('fills middle gaps when multiple existing keys leave holes', () => {
    const existing = {
      schema_defs: { schema_1: S1, schema_3: S2 },
      schemas_by_path: {},
    };
    const result = buildDatasetMetadata([{ path: 'a.jsonl', schema: S3 }], existing);
    // schema_2 is the lowest unused integer; new schema lands there.
    expect(Object.keys(result.schema_defs ?? {}).sort()).toEqual([
      'schema_1',
      'schema_2',
      'schema_3',
    ]);
    expect(result.schema_defs?.schema_2).toEqual(S3);
  });

  // Regression tests for the inline-existing-schema merge bug
  // (Codex adversarial review, dedupe.ts canonicalToKey seeding).
  describe('inline existing schemas are preserved across merge', () => {
    it('promotes existing inline default into schema_defs when a new file diverges', () => {
      const existing = { schema: S1 };
      const result = buildDatasetMetadata([{ path: 'new.jsonl', schema: S2 }], existing);

      // Top-level schema must reference a real def, not undefined.
      expect(result.schema).toBeDefined();
      expect(typeof result.schema).toBe('string');
      const topKey = result.schema as string;
      expect(result.schema_defs?.[topKey]).toEqual(S1);

      // Both schemas survive the merge.
      const defValues = Object.values(result.schema_defs ?? {});
      expect(defValues).toContainEqual(S1);
      expect(defValues).toContainEqual(S2);

      // Divergent new path resolves to a valid def key.
      const newPathKey = result.schemas_by_path?.['new.jsonl'];
      expect(typeof newPathKey).toBe('string');
      expect(result.schema_defs?.[newPathKey as string]).toEqual(S2);
      expect(newPathKey).not.toBe(topKey);
    });

    it('collapses to inline when existing inline default matches every new file', () => {
      const existing = { schema: S1 };
      const result = buildDatasetMetadata(
        [
          { path: 'a.jsonl', schema: S1 },
          { path: 'b.jsonl', schema: S1 },
        ],
        existing
      );
      expect(result).toEqual({ schema: S1, schema_defs: {}, schemas_by_path: {} });
    });

    it('promotes inline schemas_by_path values into schema_defs on merge (no schema lost)', () => {
      const existing = {
        schema: S1,
        schemas_by_path: { 'old.jsonl': S2 },
      };
      const result = buildDatasetMetadata([{ path: 'new.jsonl', schema: S3 }], existing);

      const defValues = Object.values(result.schema_defs ?? {});
      expect(defValues).toContainEqual(S1);
      expect(defValues).toContainEqual(S2);
      expect(defValues).toContainEqual(S3);

      // The previously-inline path is now a string ref.
      expect(typeof result.schemas_by_path?.['old.jsonl']).toBe('string');
      // The new divergent path is a string ref.
      expect(typeof result.schemas_by_path?.['new.jsonl']).toBe('string');
      // Top-level is the existing default (S1) as a ref.
      const topKey = result.schema as string;
      expect(result.schema_defs?.[topKey]).toEqual(S1);
    });
  });
});
