// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildAnonymizerRecord,
  buildReplacedEntities,
  outputColumn,
  parseEntities,
  parseReplacements,
  toSegments,
  toUtf16Offsets,
} from '@studio/components/AnonymizerRecordView/parse';
import {
  entities,
  ORIGINAL,
  replacements,
  REPLACED,
  traceRow,
} from '@studio/components/AnonymizerRecordView/testFixtures';

describe('parseEntities', () => {
  it('reads spans out of the wrapped entity list', () => {
    expect(parseEntities(traceRow.final_entities)).toEqual(entities);
  });

  it('decodes cells that arrive as JSON strings', () => {
    expect(parseEntities(JSON.stringify(traceRow.final_entities))).toEqual(entities);
  });

  it('reads a bare entity list, the shape job artifacts persist', () => {
    expect(parseEntities(traceRow.final_entities.entities)).toEqual(entities);
    expect(parseEntities(JSON.stringify(traceRow.final_entities.entities))).toEqual(entities);
  });

  it('drops entries missing positions', () => {
    expect(parseEntities({ entities: [{ value: 'x', label: 'y' }] })).toEqual([]);
  });

  it('returns nothing for unusable cells', () => {
    expect(parseEntities(undefined)).toEqual([]);
    expect(parseEntities('not json')).toEqual([]);
    expect(parseEntities({ entities: 'nope' })).toEqual([]);
  });
});

describe('parseReplacements', () => {
  it('reads the replacement triples', () => {
    expect(parseReplacements(traceRow._replacement_map)).toEqual(replacements);
  });

  it('drops entries that are not all strings', () => {
    expect(
      parseReplacements({ replacements: [{ original: 'a', label: 'b', synthetic: 7 }] })
    ).toEqual([]);
  });
});

describe('toSegments', () => {
  it('splits text around the tagged spans', () => {
    expect(toSegments(ORIGINAL, entities)).toEqual([
      { text: 'Bobby', label: 'first_name' },
      { text: ', a ' },
      { text: '40', label: 'age' },
      { text: '-year-old veterinarian.' },
    ]);
  });

  it('returns the whole text when nothing was detected', () => {
    expect(toSegments(ORIGINAL, [])).toEqual([{ text: ORIGINAL }]);
  });

  it('skips spans that overlap an earlier one or run past the text', () => {
    const overlapping = [
      { value: 'Bobby', label: 'first_name', start: 0, end: 5 },
      { value: 'obby', label: 'first_name', start: 1, end: 5 },
      { value: 'far', label: 'age', start: 900, end: 903 },
    ];
    expect(toSegments(ORIGINAL, overlapping)).toEqual([
      { text: 'Bobby', label: 'first_name' },
      { text: ', a 40-year-old veterinarian.' },
    ]);
  });
});

describe('buildReplacedEntities', () => {
  it('positions each entity at its synthetic value in the replaced text', () => {
    expect(buildReplacedEntities(entities, replacements, ORIGINAL, REPLACED)).toEqual([
      { value: 'Teddy', label: 'first_name', start: 0, end: 5 },
      { value: '45', label: 'age', start: 9, end: 11 },
    ]);
  });

  it('tracks offsets when a replacement changes length', () => {
    const replaced = 'Bartholomew, a 45-year-old veterinarian.';
    const longer = [
      { original: 'Bobby', label: 'first_name', synthetic: 'Bartholomew' },
      ...replacements.slice(1),
    ];
    expect(buildReplacedEntities(entities, longer, ORIGINAL, replaced)).toEqual([
      { value: 'Bartholomew', label: 'first_name', start: 0, end: 11 },
      { value: '45', label: 'age', start: 15, end: 17 },
    ]);
  });

  it('falls back to the original span when the synthetic value is absent', () => {
    const unchanged = 'Bobby, a 45-year-old veterinarian.';
    expect(buildReplacedEntities(entities, replacements, ORIGINAL, unchanged)).toEqual([
      { value: 'Bobby', label: 'first_name', start: 0, end: 5 },
      { value: '45', label: 'age', start: 9, end: 11 },
    ]);
  });

  it('skips an identical literal that the replacement did not produce', () => {
    const source = 'Teddy met Bobby';
    const output = 'Teddy met Teddy';
    const bobby = [{ value: 'Bobby', label: 'first_name', start: 10, end: 15 }];

    expect(buildReplacedEntities(bobby, replacements, source, output)).toEqual([
      { value: 'Teddy', label: 'first_name', start: 10, end: 15 },
    ]);
  });

  it('skips an identical literal that a length-changing replacement moved past', () => {
    const source = 'Alexander knows Teddy and Bobby';
    const output = 'Al knows Teddy and Teddy';
    const shifted = [
      { value: 'Alexander', label: 'first_name', start: 0, end: 9 },
      { value: 'Bobby', label: 'first_name', start: 26, end: 31 },
    ];
    const shorter = [
      { original: 'Alexander', label: 'first_name', synthetic: 'Al' },
      { original: 'Bobby', label: 'first_name', synthetic: 'Teddy' },
    ];

    expect(buildReplacedEntities(shifted, shorter, source, output)).toEqual([
      { value: 'Al', label: 'first_name', start: 0, end: 2 },
      { value: 'Teddy', label: 'first_name', start: 19, end: 24 },
    ]);
  });

  it('matches case-insensitively when the map key differs in case', () => {
    const mixedCase = [{ original: 'bobby', label: 'first_name', synthetic: 'Teddy' }];
    expect(buildReplacedEntities([entities[0]], mixedCase, ORIGINAL, REPLACED)).toEqual([
      { value: 'Teddy', label: 'first_name', start: 0, end: 5 },
    ]);
  });
});

describe('toUtf16Offsets', () => {
  it('shifts offsets past a non-BMP character', () => {
    const text = '😀 Alice';
    const detected = [{ value: 'Alice', label: 'first_name', start: 2, end: 7 }];

    expect(toUtf16Offsets(text, detected)).toEqual([
      { value: 'Alice', label: 'first_name', start: 3, end: 8 },
    ]);
    expect(toSegments(text, toUtf16Offsets(text, detected))).toEqual([
      { text: '😀 ' },
      { text: 'Alice', label: 'first_name' },
    ]);
  });

  it('leaves offsets alone when the text is all BMP', () => {
    expect(toUtf16Offsets(ORIGINAL, entities)).toBe(entities);
  });

  it('drops spans that run past the last code point', () => {
    const text = '😀 Alice';

    expect(
      toUtf16Offsets(text, [{ value: 'Alice', label: 'first_name', start: 2, end: 900 }])
    ).toEqual([]);
  });

  it('falls back to the replacement map when every converted span is dropped', () => {
    const record = buildAnonymizerRecord(
      {
        biography: '😀 Bobby is a veterinarian.',
        biography_replaced: '😀 Teddy is a veterinarian.',
        final_entities: {
          entities: [{ value: 'Bobby', label: 'first_name', start_position: 2, end_position: 900 }],
        },
        _replacement_map: { replacements: [replacements[0]] },
      },
      'biography'
    );

    expect(record.originalSegments).toEqual([
      { text: '😀 ' },
      { text: 'Bobby', label: 'first_name' },
      { text: ' is a veterinarian.' },
    ]);
  });
});

describe('outputColumn', () => {
  it('finds the replace output', () => {
    expect(outputColumn(traceRow, 'biography')).toBe('biography_replaced');
  });

  it('prefers the rewrite output', () => {
    expect(outputColumn({ text_rewritten: '', text_replaced: '' }, 'text')).toBe('text_rewritten');
  });

  it('returns nothing when neither output is present', () => {
    expect(outputColumn(traceRow, 'other')).toBeUndefined();
  });
});

describe('buildAnonymizerRecord', () => {
  it('builds both highlighted columns and the replacement map', () => {
    const record = buildAnonymizerRecord(traceRow, 'biography');

    expect(record.original).toBe(ORIGINAL);
    expect(record.replaced).toBe(REPLACED);
    expect(record.replacements).toEqual(replacements);
    expect(record.originalSegments).toEqual([
      { text: 'Bobby', label: 'first_name' },
      { text: ', a ' },
      { text: '40', label: 'age' },
      { text: '-year-old veterinarian.' },
    ]);
    expect(record.replacedSegments).toEqual([
      { text: 'Teddy', label: 'first_name' },
      { text: ', a ' },
      { text: '45', label: 'age' },
      { text: '-year-old veterinarian.' },
    ]);
  });

  it('derives spans from the replacement map when detection produced none', () => {
    const record = buildAnonymizerRecord(
      { biography: ORIGINAL, biography_replaced: REPLACED, _replacement_map: { replacements } },
      'biography'
    );

    expect(record.originalSegments).toEqual([
      { text: 'Bobby', label: 'first_name' },
      { text: ', a ' },
      { text: '40', label: 'age' },
      { text: '-year-old veterinarian.' },
    ]);
    expect(record.replacedSegments).toEqual([
      { text: 'Teddy', label: 'first_name' },
      { text: ', a ' },
      { text: '45', label: 'age' },
      { text: '-year-old veterinarian.' },
    ]);
  });

  it('reads the rewrite output column', () => {
    const record = buildAnonymizerRecord(
      { text: ORIGINAL, text_rewritten: 'A veterinarian in his forties.' },
      'text'
    );

    expect(record.replaced).toBe('A veterinarian in his forties.');
    expect(record.replacements).toEqual([]);
  });

  it('is empty when the record has no text', () => {
    const record = buildAnonymizerRecord({}, 'biography');

    expect(record.originalSegments).toEqual([]);
    expect(record.replacedSegments).toEqual([]);
  });
});
