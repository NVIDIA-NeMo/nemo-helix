// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Column } from '@tanstack/react-table';
import type { JSX } from 'react';

/**
 * Use this type to indicate that a value is a placeholder and should be replaced with a proper type.
 * Not meant to just replace `any` - we should use `any` in appropriate places instead of TSFixMe.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- intentional escape hatch
export type TSFixMe = any;

/**
 * Use this type to indicate that a value is intentionally `any` and should not be changed.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- intentional escape hatch
export type IntentionalAny = any;

/** The status a request might be in that should be reflected in the UI. */
export type QueryStatus = 'loading' | 'error' | 'success';

/**
 * When managing the state for a multi-select type filter we need to track multiple values - we do
 * so with a Record where the key is the value of the filter item and the value is a boolean
 * indicating whether the filter item is selected (always true in this case).
 */
export type MultiState = Record<FilterItem['value'], true>;

/**
 * When managing the state for a filter it can be a string (the value of a text filter or
 * single-select), or a MultiState for multi-select filters. If a value hasn't been applied it can
 * be `undefined`.
 */
export type FilterValue = string | MultiState | undefined;

/**
 * When defining single or multi select filters provide your options in this format. If a label is
 * not provided, the `value` will be used as the label.
 */
export interface FilterItem {
  /** Whether the filter item is disabled. @defaultValue false */
  disabled?: boolean;
  /** The label to display for the filter item. If not provided, the id will be used. */
  label?: string;
  /** The value to use when filtering. */
  value: string;
}

export type DataViewColumnFilterDef<TData> = {
  /** The label for the applied filter, will default to the Header if not provided. */
  label?: string;
  /** Whether the filter is loading. If true the filter will be put into a loading state. */
  loading?: boolean;
} & (
  | {
      /** Renders a text filter. */
      type: 'text';
      /** Placeholder for the text filter. @defaultValue "Filter" */
      placeholder?: string;
    }
  | {
      /** Renders a boolean filter. */
      type: 'boolean';
    }
  | {
      /**
       * `single-select` renders radio buttons; `multi-select` renders checkboxes.
       */
      type: 'single-select' | 'multi-select';
      /**
       * The options to display. If not provided, the table will auto-generate options from the
       * first 500 unique values from currently filtered data.
       */
      options?: FilterItem[];
      /** Customize options based on the column. */
      optionsBuilder?: (column: Column<TData>) => FilterItem[];
    }
  | {
      /** Provide a fully custom filter component. */
      type: 'custom';
      /** Labels for the applied-filter chip; `renderFilter` renders its own controls. */
      options?: FilterItem[];
      renderFilter: (props: {
        column: Column<TData>;
        setValue: (value: FilterValue) => void;
        value: FilterValue;
      }) => JSX.Element;
    }
);

/**
 * Describes how data is managed.
 *
 * - `auto` - DataView manages sorting, filtering, and pagination.
 * - `manual` - DataView does not manage sorting, filtering, or pagination.
 * - `sort-filter-only` - DataView manages sorting and filtering, but not pagination.
 */
export type WithDataViewDataMode =
  | {
      dataMode?: 'auto';
      totalCount?: never;
    }
  | {
      dataMode: 'manual' | 'sort-filter-only';
      totalCount: number | undefined;
    };

export type DataMode = NonNullable<WithDataViewDataMode['dataMode']>;
