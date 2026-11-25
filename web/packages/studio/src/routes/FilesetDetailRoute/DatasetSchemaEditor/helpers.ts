// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FileFormatType } from '@nemo/common/src/types';
import type { DatasetMetadataContent } from '@nemo/sdk/generated/platform/schema';
import { FilesUpdateFilesetMetadataBody } from '@nemo/sdk/generated/platform/zod/files/filesUpdateFilesetMetadata';
import { FORMAT_BY_EXTENSION } from '@studio/routes/FilesetDetailRoute/DatasetSchemaEditor/constants';
import {
  DEFAULT_SCHEMA_VALUE,
  SHOW_ALL_VALUE,
} from '@studio/routes/FilesetDetailRoute/DatasetSchemaEditor/SchemaSelectControl';

export function detectFormatFromPath(path: string): FileFormatType | null {
  const ext = path.split('.').pop()?.toLowerCase() ?? '';
  return FORMAT_BY_EXTENSION[ext] ?? null;
}

/** Resolve the effective JSON Schema applied to a file path, given a metadata
 *  payload. Mirrors backend resolution: explicit `schemas_by_path` mapping wins
 *  (string ref → schema_defs entry, inline object → the object itself);
 *  otherwise fall back to root `schema` (ref → schema_defs entry, inline →
 *  the object). Returns undefined when no schema applies. */
export function resolveSchemaForFile(
  metadata: DatasetMetadataContent | undefined,
  path: string
): unknown {
  if (!metadata) return undefined;
  const mapped = metadata.schemas_by_path?.[path];
  if (typeof mapped === 'string') return metadata.schema_defs?.[mapped];
  if (mapped && typeof mapped === 'object') return mapped;
  const root = metadata.schema;
  if (root === undefined || root === null) return undefined;
  if (typeof root === 'string') return metadata.schema_defs?.[root];
  return root;
}

/** Look up the JSON Schema object that backs the given dropdown selection. */
export function lookupSchemaForSelection(
  metadata: DatasetMetadataContent | undefined,
  selection: string
): Record<string, unknown> | undefined {
  if (!metadata) return undefined;
  if (selection === SHOW_ALL_VALUE) return undefined; // Show All is whole-payload, not a single schema.
  if (selection === DEFAULT_SCHEMA_VALUE) {
    const root = metadata.schema;
    if (root === undefined || root === null) return undefined;
    if (typeof root === 'string') return metadata.schema_defs?.[root];
    return root as Record<string, unknown>;
  }
  return metadata.schema_defs?.[selection];
}

/** True when a schema is an object-typed JSON Schema with a `properties` map.
 *  Those schemas are rendered "properties-only" in the editor; everything else
 *  is rendered as the whole schema object. */
export function hasPropertiesMap(schema: Record<string, unknown> | undefined): boolean {
  if (!schema) return false;
  const props = schema.properties;
  return props !== null && typeof props === 'object' && !Array.isArray(props);
}

/** Resolve what the editor should display for a given dropdown selection.
 *  For single-schema selections, this returns just the `properties` value
 *  when present (so the user edits field definitions, not the surrounding
 *  $schema / type wrapper). */
export function deriveSelectionText(
  metadata: DatasetMetadataContent | undefined,
  selection: string
): string {
  if (!metadata) return '';
  if (selection === SHOW_ALL_VALUE) return JSON.stringify(metadata, null, 2);
  const schema = lookupSchemaForSelection(metadata, selection);
  if (!schema) return '';
  if (hasPropertiesMap(schema)) {
    return JSON.stringify(schema.properties, null, 2);
  }
  return JSON.stringify(schema, null, 2);
}

/** Build the updated `metadata.dataset` payload for a single-schema edit.
 *  The `parsedEditorValue` is whatever the user typed in the editor — which
 *  is either the `properties` map (when the original schema had one) or the
 *  whole schema object. We look up the original schema to decide which case
 *  applies and rebuild the schema accordingly, preserving non-`properties`
 *  fields like `$schema`, `type`, `required`, etc. */
export function applySingleSchemaEdit(
  metadata: DatasetMetadataContent | undefined,
  selection: string,
  parsedEditorValue: Record<string, unknown>
): DatasetMetadataContent | undefined {
  const base: DatasetMetadataContent = metadata
    ? {
        schema: metadata.schema,
        schema_defs: { ...(metadata.schema_defs ?? {}) },
        schemas_by_path: { ...(metadata.schemas_by_path ?? {}) },
      }
    : { schema_defs: {}, schemas_by_path: {} };

  const original = lookupSchemaForSelection(metadata, selection);
  // If the original had a `properties` map, the editor was showing just that
  // value — re-wrap into the original shell. Otherwise the editor was
  // showing the full schema and the parsed value IS the new schema.
  const newSchema: Record<string, unknown> = hasPropertiesMap(original)
    ? { ...(original as Record<string, unknown>), properties: parsedEditorValue }
    : parsedEditorValue;

  if (selection === DEFAULT_SCHEMA_VALUE) {
    const root = base.schema;
    if (typeof root === 'string') {
      // Root is a ref to a schema_def; update that def in place.
      base.schema_defs = { ...(base.schema_defs ?? {}), [root]: newSchema };
    } else {
      // Inline (or absent) → set inline.
      base.schema = newSchema;
    }
    return base;
  }

  // selection is a schema_defs key
  base.schema_defs = { ...(base.schema_defs ?? {}), [selection]: newSchema };
  return base;
}

const DatasetMetadataContentSchema = FilesUpdateFilesetMetadataBody.shape.metadata
  .unwrap()
  .shape.dataset.unwrap();

/** Check an edited Show All payload against the API's `DatasetMetadataContent` shape. */
export function validateDatasetMetadata(
  value: Record<string, unknown>
): { valid: true; errors: [] } | { valid: false; errors: string[] } {
  const result = DatasetMetadataContentSchema.safeParse(value);
  if (result.success) return { valid: true, errors: [] };
  return {
    valid: false,
    errors: result.error.issues.map((issue) =>
      issue.path.length > 0 ? `${issue.path.join('.')}: ${issue.message}` : issue.message
    ),
  };
}
