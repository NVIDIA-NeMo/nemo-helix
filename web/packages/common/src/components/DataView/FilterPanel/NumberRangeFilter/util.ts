// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from 'react';

/**
 * Filter value emitted by the numeric range filter. Mirrors the Mongo-style
 * `$gte` / `$lte` operators used across NeMo Helix's unified filter syntax
 * (see `@nemo/common/src/api/filterOperators`). Either bound may be omitted to
 * express an open-ended range.
 */
export interface NumberRangeFilterValue {
  /** Inclusive lower bound. Omitted when the range is open on the low end. */
  $gte?: number;
  /** Inclusive upper bound. Omitted when the range is open on the high end. */
  $lte?: number;
}

interface NumberRangeColumnFilter {
  label: string;
  type: 'custom';
  filterVariant: 'numberRange';
  /** Lower bound of the slider track. Omit to hide the slider and leave the min input empty. */
  min?: number;
  /** Upper bound of the slider track. Omit to hide the slider and leave the max input empty. */
  max?: number;
  /** Stepping interval for the slider and the min/max inputs. */
  step: number;
  renderFilter: () => React.JSX.Element;
}

export interface NumberRangeFilterConfig {
  /** Lower bound of the slider track. Omit to hide the slider and leave the min input empty. */
  min?: number;
  /** Upper bound of the slider track. Omit to hide the slider and leave the max input empty. */
  max?: number;
  /** Stepping interval for the slider and the min/max inputs. @defaultValue 1 */
  step?: number;
}

/**
 * Builds a numberRange filter def for a column's `meta.filter`, stored as
 * `{ $gte, $lte }`. Supplying both `min`/`max` shows the slider; step defaults to 1.
 */
export function numberRangeFilter(
  label: string,
  { min, max, step = 1 }: NumberRangeFilterConfig = {}
): NumberRangeColumnFilter {
  return {
    label,
    type: 'custom',
    filterVariant: 'numberRange',
    min,
    max,
    step,
    // Placeholder: number-range columns are dispatched via `isNumberRangeFilter`
    // in ColumnFilterPanel, so this factory output never renders through `renderFilter`.
    // Kept as a JSX-free Fragment so this stays a `.ts` module.
    renderFilter: () => React.createElement(React.Fragment),
  };
}

/** Type guard: checks whether a filter def is a numeric range filter. */
export function isNumberRangeFilter(
  filter: { type: string } | undefined | null
): filter is NumberRangeColumnFilter {
  return (
    filter != null &&
    filter.type === 'custom' &&
    'filterVariant' in filter &&
    (filter as NumberRangeColumnFilter).filterVariant === 'numberRange'
  );
}

/**
 * Formats a `{ $gte, $lte }` numeric range into a readable string for applied
 * filter chips. Open-ended ranges render with a single comparator.
 *
 * - both bounds: `"10 – 50"`
 * - lower only: `"≥ 10"`
 * - upper only: `"≤ 50"`
 * - neither: `""`
 */
export function formatNumberRange(gte?: number, lte?: number): string {
  const format = (value: number) => value.toLocaleString();
  if (gte != null && lte != null) return `${format(gte)} – ${format(lte)}`;
  if (gte != null) return `≥ ${format(gte)}`;
  if (lte != null) return `≤ ${format(lte)}`;
  return '';
}
