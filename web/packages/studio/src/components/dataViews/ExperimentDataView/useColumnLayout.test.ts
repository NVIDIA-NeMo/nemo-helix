// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  hiddenColumnIds,
  isLayoutDirty,
  seedColumnState,
} from '@studio/components/dataViews/ExperimentDataView/useColumnLayout';

describe('hiddenColumnIds', () => {
  it('reports only the columns visibility turns off', () => {
    expect(hiddenColumnIds({ name: true, created_by: false, updated_at: false })).toEqual([
      'created_by',
      'updated_at',
    ]);
  });

  it('is empty when every recorded column is visible', () => {
    expect(hiddenColumnIds({ name: true, created_by: true })).toEqual([]);
  });
});

describe('seedColumnState', () => {
  it('applies the built-in hidden columns when nothing has been saved', () => {
    expect(seedColumnState(undefined)).toEqual({
      columnOrder: [],
      columnVisibility: { created_by: false, updated_at: false },
    });
  });

  it('applies the built-in hidden columns when the API sends null', () => {
    expect(seedColumnState(null)).toEqual({
      columnOrder: [],
      columnVisibility: { created_by: false, updated_at: false },
    });
  });

  it('treats a saved layout that hides nothing as authoritative', () => {
    // Why the field is nullable: an explicit "show everything" must survive the next load.
    expect(seedColumnState({ order: [], hidden: [] })).toEqual({
      columnOrder: [],
      columnVisibility: {},
    });
  });

  it('seeds the saved order and hidden columns', () => {
    expect(seedColumnState({ order: ['name', 'cost_usd'], hidden: ['tokens'] })).toEqual({
      columnOrder: ['name', 'cost_usd'],
      columnVisibility: { tokens: false },
    });
  });
});

describe('isLayoutDirty', () => {
  it('is clean when an untouched table sits on the defaults', () => {
    // The load-time case: the Save button must not appear before anyone has changed anything.
    const seeded = seedColumnState(undefined);
    expect(
      isLayoutDirty({
        saved: undefined,
        columnOrder: seeded.columnOrder,
        columnVisibility: seeded.columnVisibility,
      })
    ).toBe(false);
  });

  it('is clean when an untouched table sits on its saved layout', () => {
    const saved = { order: ['name', 'cost_usd'], hidden: ['tokens'] };
    const seeded = seedColumnState(saved);
    expect(
      isLayoutDirty({
        saved,
        columnOrder: seeded.columnOrder,
        columnVisibility: seeded.columnVisibility,
      })
    ).toBe(false);
  });

  it('is dirty once a column is hidden', () => {
    expect(
      isLayoutDirty({
        saved: undefined,
        columnOrder: [],
        columnVisibility: { created_by: false, updated_at: false, tokens: false },
      })
    ).toBe(true);
  });

  it('is dirty once a hidden column is shown again', () => {
    expect(
      isLayoutDirty({
        saved: { order: [], hidden: ['tokens'] },
        columnOrder: [],
        columnVisibility: { tokens: true },
      })
    ).toBe(true);
  });

  it('is dirty once columns are reordered', () => {
    expect(
      isLayoutDirty({
        saved: { order: ['name', 'cost_usd'], hidden: [] },
        columnOrder: ['cost_usd', 'name'],
        columnVisibility: {},
      })
    ).toBe(true);
  });

  it('compares hidden columns as a set, not a sequence', () => {
    // Visibility is a map, so ids arrive in toggle order; that alone is not worth a save.
    expect(
      isLayoutDirty({
        saved: { order: [], hidden: ['updated_at', 'created_by'] },
        columnOrder: [],
        columnVisibility: { created_by: false, updated_at: false },
      })
    ).toBe(false);
  });
});
