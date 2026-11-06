// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { canonicalJson } from '@nemo/common/src/utils/jsonSchema/canonical';
import type { DatasetMetadataContent } from '@nemo/sdk/generated/platform/schema';

/**
 * Build a `metadata.dataset` payload from per-file inferred schemas, optionally
 * merging with an existing payload (e.g. a fileset that already has metadata).
 *
 * Output shape rules:
 *   - Zero files and no existing  -> {}.
 *   - Zero files with existing    -> existing unchanged (deep-clone).
 *   - Single unique schema across all known schemas (new files + existing
 *     inline default + existing inline path values + existing def values) and
 *     no preserved named defs -> emit inline `schema` only.
 *   - Otherwise -> emit `schema` as a ref to the dominant def, `schema_defs`
 *     keyed by `schema_N` (lowest unused integer), and `schemas_by_path` for
 *     paths whose schema diverges from the dominant.
 *
 * When `existing` is provided:
 *   - Existing def keys are preserved.
 *   - New canonical schemas mint new `schema_N` keys.
 *   - `existing.schema` (inline or ref) is treated as the default canonical;
 *     dominant-from-new-files is only computed when existing had no default.
 *   - Inline schemas anywhere in `existing` (top-level `schema` or path values)
 *     are promoted to `schema_defs` entries when the multi-schema path is
 *     reached, so the merged payload never silently loses an existing schema.
 */

export interface PerFileInferred {
  path: string;
  schema: Record<string, unknown>;
}

const DEF_KEY_PATTERN = /^schema_(\d+)$/;

export function buildDatasetMetadata(
  perFile: PerFileInferred[],
  existing?: DatasetMetadataContent
): DatasetMetadataContent {
  if (perFile.length === 0) {
    return existing ? cloneMetadata(existing) : {};
  }

  // Fast path: no existing metadata and all new files share one canonical -> inline.
  if (!existing && allIdentical(perFile)) {
    return {
      schema: perFile[0].schema,
      schema_defs: {},
      schemas_by_path: {},
    };
  }

  // Track every canonical we've seen across new files + existing metadata.
  // `canonicalToSchema` covers EVERY known schema (so the collapse check and
  // multi-schema promotion never lose an inline schema). `canonicalToKey`
  // covers only canonicals that already have a def key (initially the
  // preserved `existing.schema_defs` keys).
  const defs: Record<string, Record<string, unknown>> = { ...(existing?.schema_defs ?? {}) };
  const canonicalToKey = new Map<string, string>();
  const canonicalToSchema = new Map<string, Record<string, unknown>>();

  for (const [key, schema] of Object.entries(defs)) {
    const canon = canonicalJson(schema);
    canonicalToKey.set(canon, key);
    canonicalToSchema.set(canon, schema);
  }

  // Register existing inline default (if present).
  if (existing?.schema && typeof existing.schema === 'object') {
    const canon = canonicalJson(existing.schema);
    if (!canonicalToSchema.has(canon)) {
      canonicalToSchema.set(canon, existing.schema as Record<string, unknown>);
    }
  }

  // Register inline schemas inside existing.schemas_by_path.
  for (const value of Object.values(existing?.schemas_by_path ?? {})) {
    if (typeof value === 'object') {
      const canon = canonicalJson(value);
      if (!canonicalToSchema.has(canon)) {
        canonicalToSchema.set(canon, value as Record<string, unknown>);
      }
    }
  }

  // Register new file schemas (preserves encounter order for tie-breaks).
  const newFileCanonicals: string[] = [];
  for (const file of perFile) {
    const canon = canonicalJson(file.schema);
    newFileCanonicals.push(canon);
    if (!canonicalToSchema.has(canon)) {
      canonicalToSchema.set(canon, file.schema);
    }
  }

  // Resolve all known paths (existing + new) to a canonical.
  const pathToCanonical = new Map<string, string>();
  for (const [path, value] of Object.entries(existing?.schemas_by_path ?? {})) {
    const canon = resolvePathValueCanonical(value, defs);
    if (canon) pathToCanonical.set(path, canon);
  }
  for (let i = 0; i < perFile.length; i++) {
    pathToCanonical.set(perFile[i].path, newFileCanonicals[i]);
  }

  // Pick top-level canonical: existing default wins, else dominant from new files.
  const existingDefaultCanonical = resolveExistingDefaultCanonical(existing);
  const topCanonical = existingDefaultCanonical ?? dominantCanonical(newFileCanonicals);

  // Single-schema collapse: every known schema across new + existing agrees
  // AND there were no preserved named defs (which a user may rely on by name).
  const existingDefCount = Object.keys(existing?.schema_defs ?? {}).length;
  if (canonicalToSchema.size === 1 && existingDefCount === 0 && topCanonical) {
    const onlySchema = canonicalToSchema.get(topCanonical)!;
    return {
      schema: onlySchema,
      schema_defs: {},
      schemas_by_path: {},
    };
  }

  // Multi-schema path: mint def keys for any canonical that doesn't have one
  // yet. This promotes existing inline defaults and inline path values into
  // `schema_defs` so the merged payload preserves every schema we saw.
  for (const canon of canonicalToSchema.keys()) {
    if (!canonicalToKey.has(canon)) {
      const key = nextDefKey(defs);
      defs[key] = canonicalToSchema.get(canon)!;
      canonicalToKey.set(canon, key);
    }
  }

  // Multi-schema output: schema = ref to top, schemas_by_path = divergent paths.
  const topKey = topCanonical ? canonicalToKey.get(topCanonical) : undefined;
  const byPath: Record<string, string> = {};
  for (const [path, canon] of pathToCanonical) {
    if (canon !== topCanonical) {
      byPath[path] = canonicalToKey.get(canon)!;
    }
  }

  return {
    schema: topKey,
    schema_defs: defs,
    schemas_by_path: byPath,
  };
}

function allIdentical(perFile: PerFileInferred[]): boolean {
  const first = canonicalJson(perFile[0].schema);
  return perFile.every((f) => canonicalJson(f.schema) === first);
}

function dominantCanonical(canonicals: string[]): string | undefined {
  if (canonicals.length === 0) return undefined;
  const counts = new Map<string, number>();
  for (const c of canonicals) {
    counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  // Tie-break: first encountered.
  let best: string | undefined;
  let max = 0;
  for (const c of canonicals) {
    const n = counts.get(c)!;
    if (n > max) {
      max = n;
      best = c;
    }
  }
  return best;
}

function resolveExistingDefaultCanonical(
  existing: DatasetMetadataContent | undefined
): string | undefined {
  if (!existing?.schema) return undefined;
  if (typeof existing.schema === 'string') {
    const def = existing.schema_defs?.[existing.schema];
    return def ? canonicalJson(def) : undefined;
  }
  return canonicalJson(existing.schema);
}

function resolvePathValueCanonical(
  value: Record<string, unknown> | string,
  defs: Record<string, Record<string, unknown>>
): string | undefined {
  if (typeof value === 'string') {
    const def = defs[value];
    return def ? canonicalJson(def) : undefined;
  }
  return canonicalJson(value);
}

function nextDefKey(defs: Record<string, unknown>): string {
  // Lowest unused positive integer: fills gaps (e.g. {schema_1, schema_3} -> schema_2)
  // so the schema_N namespace stays compact across edits.
  const used = new Set<number>();
  for (const key of Object.keys(defs)) {
    const match = key.match(DEF_KEY_PATTERN);
    if (match) used.add(Number(match[1]));
  }
  let n = 1;
  while (used.has(n)) n++;
  return `schema_${n}`;
}

function cloneMetadata(meta: DatasetMetadataContent): DatasetMetadataContent {
  // structuredClone produces an independent deep copy. DatasetMetadataContent
  // is plain JSON (dict / string / nested dict), so the structured-clone
  // algorithm covers every field shape we accept.
  return structuredClone(meta);
}
