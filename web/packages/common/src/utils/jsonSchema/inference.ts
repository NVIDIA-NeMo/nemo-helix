// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isPlainObject } from '@nemo/common/src/utils/isPlainObject';

/**
 * JSON Schema inference from a sample value (e.g. the 0th row of a JSONL file).
 *
 * Conservative defaults:
 *   - `required` is omitted: one observed row can only say "these keys exist",
 *     not "these keys are mandatory."
 *   - `null` values emit `{}` (no constraint) rather than `{ type: "null" }`, so
 *     future rows that happen to have a value for the same key still validate.
 *   - Arrays infer `items.type` only when every element is the same primitive.
 *     Mixed primitives, empty arrays, or arrays of objects emit `items: {}`.
 *   - Recursion is capped at `INFERENCE_MAX_DEPTH` to keep deeply nested rows
 *     from blowing up the inferred document.
 */

export const INFERENCE_MAX_DEPTH = 8;

const JSON_SCHEMA_DRAFT = 'https://json-schema.org/draft/2020-12/schema';

export interface InferOptions {
  /** Override the recursion cap. Defaults to {@link INFERENCE_MAX_DEPTH}. */
  maxDepth?: number;
}

export type JsonSchema = Record<string, unknown>;

/** Top-level entry point. Always stamps the document with `$schema`. */
export function inferJsonSchema(value: unknown, opts?: InferOptions): JsonSchema {
  const maxDepth = opts?.maxDepth ?? INFERENCE_MAX_DEPTH;
  const inner = inferValue(value, 0, maxDepth);
  return { $schema: JSON_SCHEMA_DRAFT, ...inner };
}

function inferValue(value: unknown, depth: number, maxDepth: number): JsonSchema {
  if (depth >= maxDepth) {
    if (Array.isArray(value)) return { type: 'array', items: {} };
    if (isPlainObject(value)) return { type: 'object' };
    return {};
  }
  if (value === null || value === undefined) return {};
  if (typeof value === 'string') return { type: 'string' };
  if (typeof value === 'boolean') return { type: 'boolean' };
  if (typeof value === 'number') {
    return { type: Number.isInteger(value) ? 'integer' : 'number' };
  }
  if (Array.isArray(value)) {
    return { type: 'array', items: inferArrayItems(value, depth + 1, maxDepth) };
  }
  if (isPlainObject(value)) {
    const properties: Record<string, JsonSchema> = {};
    for (const key of Object.keys(value)) {
      properties[key] = inferValue(value[key], depth + 1, maxDepth);
    }
    return { type: 'object', properties };
  }
  return {};
}

function inferArrayItems(arr: unknown[], depth: number, maxDepth: number): JsonSchema {
  if (arr.length === 0) return {};
  const types = arr.map(getPrimitiveType);
  const first = types[0];
  if (first !== null && types.every((t) => t === first)) {
    return { type: first };
  }
  // Homogeneous object case: every element is a plain object. Infer items
  // from the first element so common shapes like message arrays produce a
  // structured items schema (`{type: object, properties: {...}}`) instead
  // of a bare `{}`. First-element inference may miss keys that only appear
  // in later items; the user can refine in the editor.
  if (arr.every(isPlainObject)) {
    return inferValue(arr[0], depth, maxDepth);
  }
  // Mixed primitives, arrays-of-arrays, or null mixed in: emit an
  // unconstrained items schema. Caller can tighten manually.
  return {};
}

function getPrimitiveType(value: unknown): 'string' | 'boolean' | 'integer' | 'number' | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string') return 'string';
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number';
  return null;
}
