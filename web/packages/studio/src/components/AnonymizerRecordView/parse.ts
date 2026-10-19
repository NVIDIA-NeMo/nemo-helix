// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AnonymizerEntity,
  AnonymizerRecord,
  EntityReplacement,
  TextSegment,
} from '@studio/components/AnonymizerRecordView/types';
import { asRecord } from '@studio/util/guards';

export const DETECTED_ENTITIES_COLUMN = '_detected_entities';
export const FINAL_ENTITIES_COLUMN = 'final_entities';
export const REPLACEMENT_MAP_COLUMN = '_replacement_map';

/** Rewrite writes `<column>_rewritten`; the replace strategies write `<column>_replaced`. */
export const REWRITTEN_SUFFIX = '_rewritten';
export const REPLACED_SUFFIX = '_replaced';
export const OUTPUT_SUFFIXES = [REWRITTEN_SUFFIX, REPLACED_SUFFIX] as const;

/** Trace cells arrive either already decoded or as a JSON string, depending on the writer. */
const decodeCell = (value: unknown): unknown => {
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return undefined;
  }
};

/** The job artifact persists these columns as a bare array; the live preview wraps them. */
const wrappedList = (cell: unknown, key: string): unknown[] => {
  const decoded = decodeCell(cell);
  if (Array.isArray(decoded)) return decoded;
  const entries = asRecord(decoded)?.[key];
  return Array.isArray(entries) ? entries : [];
};

const toEntity = (entry: unknown): AnonymizerEntity | undefined => {
  const row = asRecord(entry);
  if (!row) return undefined;
  const { value, label, start_position: start, end_position: end } = row;
  if (typeof label !== 'string' || typeof start !== 'number' || typeof end !== 'number') {
    return undefined;
  }
  return { value: typeof value === 'string' ? value : '', label, start, end };
};

export const parseEntities = (cell: unknown): AnonymizerEntity[] =>
  wrappedList(cell, 'entities').flatMap((entry) => {
    const entity = toEntity(entry);
    return entity ? [entity] : [];
  });

export const parseReplacements = (cell: unknown): EntityReplacement[] =>
  wrappedList(cell, 'replacements').flatMap((entry) => {
    const row = asRecord(entry);
    const { original, label, synthetic } = row ?? {};
    return typeof original === 'string' &&
      typeof label === 'string' &&
      typeof synthetic === 'string'
      ? [{ original, label, synthetic }]
      : [];
  });

const byPosition = (a: AnonymizerEntity, b: AnonymizerEntity): number =>
  a.start - b.start || a.end - b.end;

const SURROGATE_PAIR = /[\uD800-\uDBFF][\uDC00-\uDFFF]/;

/** Detection counts code points the way Python does; string indices here are UTF-16 units. */
export const toUtf16Offsets = (
  text: string,
  entities: readonly AnonymizerEntity[]
): readonly AnonymizerEntity[] => {
  if (!entities.length || !SURROGATE_PAIR.test(text)) return entities;

  const unitForCodePoint: number[] = [];
  for (let index = 0; index < text.length; ) {
    unitForCodePoint.push(index);
    index += (text.codePointAt(index) ?? 0) > 0xffff ? 2 : 1;
  }
  unitForCodePoint.push(text.length);

  // Dropping rather than clamping: an offset past the last code point would map onto
  // text.length and slip through the bounds check in toSegments.
  return entities.flatMap((entity) => {
    const start = unitForCodePoint[entity.start];
    const end = unitForCodePoint[entity.end];
    return start === undefined || end === undefined ? [] : [{ ...entity, start, end }];
  });
};

export const toSegments = (text: string, entities: readonly AnonymizerEntity[]): TextSegment[] => {
  if (!entities.length) return text ? [{ text }] : [];

  const segments: TextSegment[] = [];
  let cursor = 0;
  for (const entity of [...entities].sort(byPosition)) {
    if (entity.start < cursor || entity.end <= entity.start || entity.end > text.length) continue;
    if (entity.start > cursor) segments.push({ text: text.slice(cursor, entity.start) });
    segments.push({ text: text.slice(entity.start, entity.end), label: entity.label });
    cursor = entity.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor) });
  return segments;
};

interface SyntheticLookups {
  readonly byValueLabel: Map<string, string>;
  readonly byValue: Map<string, string>;
  readonly byValueLabelLower: Map<string, string>;
  readonly byValueLower: Map<string, string>;
}

// Label first so the newline separator stays unambiguous — labels never contain one, values may.
const lookupKey = (value: string, label: string): string => `${label}\n${value}`;

const buildSyntheticLookups = (replacements: readonly EntityReplacement[]): SyntheticLookups => {
  const lookups: SyntheticLookups = {
    byValueLabel: new Map(),
    byValue: new Map(),
    byValueLabelLower: new Map(),
    byValueLower: new Map(),
  };
  for (const { original, label, synthetic } of replacements) {
    const lower = original.toLowerCase();
    lookups.byValueLabel.set(lookupKey(original, label), synthetic);
    lookups.byValue.set(original, synthetic);
    lookups.byValueLabelLower.set(lookupKey(lower, label), synthetic);
    lookups.byValueLower.set(lower, synthetic);
  }
  return lookups;
};

/** Exact value+label, then value only, then both again case-insensitively. */
const resolveSynthetic = (entity: AnonymizerEntity, lookups: SyntheticLookups): string => {
  const lower = entity.value.toLowerCase();
  return (
    lookups.byValueLabel.get(lookupKey(entity.value, entity.label)) ??
    lookups.byValue.get(entity.value) ??
    lookups.byValueLabelLower.get(lookupKey(lower, entity.label)) ??
    lookups.byValueLower.get(lower) ??
    entity.value
  );
};

/** Text between entities is untouched, so the gap since the last match predicts this one. */
const locate = (text: string, needle: string, expected: number, searchFrom: number): number => {
  if (!needle) return -1;
  if (expected >= searchFrom && text.startsWith(needle, expected)) return expected;
  return text.indexOf(needle, searchFrom);
};

/** Aligns each span through the unchanged gap, then falls back to a forward search. */
export const buildReplacedEntities = (
  originalEntities: readonly AnonymizerEntity[],
  replacements: readonly EntityReplacement[],
  originalText: string,
  replacedText: string
): AnonymizerEntity[] => {
  const lookups = buildSyntheticLookups(replacements);
  const replaced: AnonymizerEntity[] = [];
  let originalCursor = 0;
  let searchFrom = 0;

  for (const entity of [...originalEntities].sort(byPosition)) {
    const { start, end, label } = entity;
    if (start < originalCursor || end <= start || end > originalText.length) continue;

    const expected = searchFrom + (start - originalCursor);
    const originalSpan = originalText.slice(start, end);
    let synthetic = resolveSynthetic(entity, lookups);
    let position = locate(replacedText, synthetic, expected, searchFrom);
    if (position < 0 && synthetic !== originalSpan) {
      position = locate(replacedText, originalSpan, expected, searchFrom);
      if (position >= 0) synthetic = originalSpan;
    }

    originalCursor = end;
    if (position < 0) continue;

    replaced.push({
      value: replacedText.slice(position, position + synthetic.length),
      label,
      start: position,
      end: position + synthetic.length,
    });
    searchFrom = position + synthetic.length;
  }

  return replaced;
};

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Fallback for when detection produced no spans but the replacement map did. */
const entitiesByCaseInsensitiveSearch = (
  replacements: readonly EntityReplacement[],
  text: string
): AnonymizerEntity[] =>
  replacements
    .flatMap(({ original, label }) => {
      if (!original || !label) return [];
      return [...text.matchAll(new RegExp(escapeRegExp(original), 'gi'))].map((match) => ({
        value: match[0],
        label,
        start: match.index,
        end: match.index + match[0].length,
      }));
    })
    .sort(byPosition);

const entitiesBySyntheticSearch = (
  replacements: readonly EntityReplacement[],
  text: string
): AnonymizerEntity[] =>
  replacements
    .flatMap(({ synthetic, label }) => {
      if (!synthetic || !label) return [];
      return [...text.matchAll(new RegExp(escapeRegExp(synthetic), 'g'))].map((match) => ({
        value: synthetic,
        label,
        start: match.index,
        end: match.index + synthetic.length,
      }));
    })
    .sort(byPosition);

const asText = (value: unknown): string => (typeof value === 'string' ? value : '');

export const outputColumn = (
  row: Record<string, unknown>,
  textColumn: string
): string | undefined =>
  OUTPUT_SUFFIXES.map((suffix) => `${textColumn}${suffix}`).find((column) => column in row);

export const buildAnonymizerRecord = (
  row: Record<string, unknown>,
  textColumn: string
): AnonymizerRecord => {
  const original = asText(row[textColumn]);
  const outputKey = outputColumn(row, textColumn);
  const replaced = outputKey ? asText(row[outputKey]) : '';
  const replacements = parseReplacements(row[REPLACEMENT_MAP_COLUMN]);

  const detected =
    FINAL_ENTITIES_COLUMN in row
      ? parseEntities(row[FINAL_ENTITIES_COLUMN])
      : parseEntities(row[DETECTED_ENTITIES_COLUMN]);
  const converted = toUtf16Offsets(original, detected);
  const originalEntities = converted.length
    ? converted
    : entitiesByCaseInsensitiveSearch(replacements, original);

  const derived = buildReplacedEntities(originalEntities, replacements, original, replaced);
  const replacedEntities = derived.length
    ? derived
    : entitiesBySyntheticSearch(replacements, replaced);

  return {
    original,
    replaced,
    originalSegments: toSegments(original, originalEntities),
    replacedSegments: toSegments(replaced, replacedEntities),
    replacements,
  };
};
