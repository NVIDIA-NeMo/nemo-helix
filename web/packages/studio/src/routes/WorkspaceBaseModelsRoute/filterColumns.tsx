// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntityFilterInput } from '@nemo/common/src/api/entity-store/useBaseModels';
import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import type * as DataView from '@nemo/common/src/components/DataView/internal';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { CUSTOMIZER_ENABLED } from '@studio/constants/environment';
import { ModelProviderFilterField } from '@studio/routes/WorkspaceBaseModelsRoute/ModelProviderFilterField';
import type { ComponentProps } from 'react';

export const CUSTOMIZABLE_FILTER_ID = 'customizable';
export const FINE_TUNABLE_KEY = 'fine_tunable';

/**
 * One "LoRA" filter for two API fields: `adapters` (trained adapters are attached to the model)
 * and `lora_enabled` (its latest deployment config can load adapters). Single-select, since the
 * API can only AND them (`lora_enabled` must be a top-level equality), not OR them.
 */
const LORA_FILTER_ID = 'lora';
const LORA_ADAPTERS = 'adapters';
const LORA_ENABLED = 'lora_enabled';
// Adapters come from Customizer runs, so hide that option while it's disabled.
const getLoraOptions = () => [
  ...(CUSTOMIZER_ENABLED ? [{ value: LORA_ADAPTERS, label: 'Has adapters' }] : []),
  { value: LORA_ENABLED, label: 'Serving enabled' },
];

/**
 * Whether toModelEntityFilter applies this filter. Shared URLs can carry ones it ignores (e.g.
 * Customizer-only filters while Customizer is disabled); drop those so no stale chip shows.
 */
export const isSupportedFilter = ({ id, value }: { id: string; value: unknown }) => {
  if (id === CUSTOMIZABLE_FILTER_ID) return CUSTOMIZER_ENABLED;
  if (id === LORA_FILTER_ID) return getLoraOptions().some((option) => option.value === value);
  return true;
};

const toOptionalString = (value: unknown) =>
  typeof value === 'string' && value ? value : undefined;

/**
 * Column definitions used solely for filter metadata. The columns are never rendered as a table;
 * they only provide `meta.filter` config so that DataView's ColumnFilterPanel and
 * StudioAppliedFilters components can render and manage the filter UI.
 *
 * The filter panel renders them in this order: LoRA (adapters, serving), family, provider, dates.
 */
export const makeFilterColumns =
  (workspace: string): ComponentProps<typeof DataView.Root<ModelEntity>>['makeColumns'] =>
  ({ accessor }) => [
    // Set by the toolbar's Fine-tunable checkbox; kept out of the panel, but labels its chip.
    ...(CUSTOMIZER_ENABLED
      ? [
          accessor(() => '', {
            id: CUSTOMIZABLE_FILTER_ID,
            header: 'Fine-tunable',
            enableSorting: false,
            enableColumnFilter: false,
            meta: {
              filter: {
                type: 'multi-select',
                label: 'Fine-tunable',
                options: [{ value: FINE_TUNABLE_KEY, label: 'Fine-tunable' }],
              },
            },
          }),
        ]
      : []),
    accessor(() => '', {
      id: LORA_FILTER_ID,
      header: 'LoRA',
      enableSorting: false,
      meta: { filter: { type: 'single-select', label: 'LoRA', options: getLoraOptions() } },
    }),
    // `spec.family` is only set on models with a checkpoint fileset, once spec analysis has run.
    accessor(() => '', {
      id: 'family',
      header: 'Family',
      enableSorting: false,
      meta: {
        filter: { type: 'text', label: 'Family', placeholder: 'e.g. llama (checkpoint models)' },
      },
    }),
    accessor(() => '', {
      id: 'model_providers',
      header: 'Model Provider',
      enableSorting: false,
      meta: {
        filter: {
          type: 'custom',
          label: 'Model Provider',
          renderFilter: ({ setValue, value }) => (
            <ModelProviderFilterField
              workspace={workspace}
              value={toOptionalString(value)}
              onValueChange={setValue}
            />
          ),
        },
      },
    }),
    accessor('created_at', {
      id: 'created_at',
      header: 'Created',
      enableSorting: false,
      meta: { filter: dateTimeFilter('Created At') },
    }),
    accessor('updated_at', {
      id: 'updated_at',
      header: 'Updated',
      enableSorting: false,
      meta: { filter: dateTimeFilter('Updated At') },
    }),
  ];

/**
 * Builds the Models API filter from DataView column filter state and the name search:
 * Fine-tunable becomes `fileset: true`, the LoRA option becomes a boolean, and `family` / name become
 * substring matches.
 */
export const toModelEntityFilter = (
  columnFilters: Record<string, unknown> | undefined,
  searchText?: string
): ModelEntityFilterInput | undefined => {
  const {
    [CUSTOMIZABLE_FILTER_ID]: customizable,
    family,
    [LORA_FILTER_ID]: lora,
    ...filter
  } = columnFilters ?? {};
  // Fine-tunable means "has a checkpoint fileset"; ignored while Customizer is disabled.
  if (CUSTOMIZER_ENABLED && customizable && Object.keys(customizable).length > 0) {
    filter.fileset = true;
  }
  if (CUSTOMIZER_ENABLED && lora === LORA_ADAPTERS) filter.adapters = true;
  if (lora === LORA_ENABLED) filter.lora_enabled = true;
  const familyText = toOptionalString(family)?.trim();
  if (familyText) filter.family = { $like: familyText };
  const name = searchText?.trim();
  if (name) filter.name = { $like: name };

  return Object.keys(filter).length > 0 ? (filter as ModelEntityFilterInput) : undefined;
};
