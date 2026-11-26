// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  toModelCapabilities,
  toModelModalities,
  toModelSizeBuckets,
  type ModelFilterControls,
  type ModelFilterOption,
} from '@nemo/common/src/api/models/modelFilters';
import { Grid, Select, Stack, type SelectEntry } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

const CAPABILITY_PREFIX = 'capability:';
const INPUT_PREFIX = 'input:';

interface ModelFilterSelectProps {
  name: string;
  label: string;
  values?: readonly string[];
  options: ModelFilterOption[];
  items?: SelectEntry[];
  placeholder: string;
  listSelected?: boolean;
  disabled?: boolean;
  onChange: (values: string[]) => void;
}

const selectionSummary = (
  selected: readonly string[],
  options: ModelFilterOption[],
  placeholder: string,
  listSelected: boolean
): string => {
  if (selected.length === 0) {
    return placeholder;
  }
  const labels = selected.map(
    (value) => options.find((option) => option.value === value)?.label ?? value
  );
  if (labels.length === 1 || listSelected) {
    return labels.join(', ');
  }
  return `${placeholder} · ${labels.length}`;
};

const toItems = (options: ModelFilterOption[]) =>
  options.map((option) => ({ value: option.value, children: option.label }));

const withPrefix = (prefix: string, options: ModelFilterOption[]): ModelFilterOption[] =>
  options.map((option) => ({ ...option, value: `${prefix}${option.value}` }));

const withoutPrefix = (prefix: string, values: readonly string[]): string[] =>
  values.filter((value) => value.startsWith(prefix)).map((value) => value.slice(prefix.length));

const ModelFilterSelect: FC<ModelFilterSelectProps> = ({
  name,
  label,
  values = [],
  options,
  items,
  placeholder,
  listSelected = false,
  disabled,
  onChange,
}) => (
  <Select
    className="min-w-0"
    size="small"
    multiple
    placeholder={placeholder}
    value={[...values]}
    dismissible
    disabled={disabled}
    onValueChange={onChange}
    renderValue={(selected) => selectionSummary(selected, options, placeholder, listSelected)}
    items={items ?? toItems(options)}
    attributes={{
      SelectTrigger: {
        'aria-label': label,
        ['data-testid' as never]: `model-filter-${name}`,
      },
    }}
  />
);

export const ModelDropdownFilters: FC<ModelFilterControls> = ({
  values,
  onChange,
  providerOptions,
  capabilityOptions,
  inputModalityOptions,
  sizeOptions,
  providersLoading,
}) => {
  const prefixedCapabilities = withPrefix(CAPABILITY_PREFIX, capabilityOptions);
  const prefixedInputs = withPrefix(INPUT_PREFIX, inputModalityOptions);

  return (
    <Stack gap="density-sm" className="px-2 pb-2 w-full" data-testid="model-select-v2-filters">
      <Grid cols={2} gap="density-sm">
        <ModelFilterSelect
          name="providers"
          label="Filter by provider"
          placeholder={providersLoading ? 'Loading...' : 'Provider'}
          values={values.providers}
          options={providerOptions}
          disabled={!values.providers?.length && providerOptions.length === 0}
          onChange={(providers) => onChange({ ...values, providers })}
        />
        <ModelFilterSelect
          name="sizes"
          label="Filter by size"
          placeholder="Size"
          values={values.sizes}
          options={sizeOptions}
          onChange={(next) => onChange({ ...values, sizes: toModelSizeBuckets(next) })}
        />
      </Grid>
      <ModelFilterSelect
        name="capabilities"
        label="Filter by capability"
        placeholder="Capabilities"
        listSelected
        values={[
          ...(values.capabilities ?? []).map((value) => `${CAPABILITY_PREFIX}${value}`),
          ...(values.inputModalities ?? []).map((value) => `${INPUT_PREFIX}${value}`),
        ]}
        options={[...prefixedCapabilities, ...prefixedInputs]}
        items={[
          { kind: 'section', slotHeading: 'Can do', items: toItems(prefixedCapabilities) },
          { kind: 'section', slotHeading: 'Accepts', items: toItems(prefixedInputs) },
        ]}
        onChange={(next) =>
          onChange({
            ...values,
            capabilities: toModelCapabilities(withoutPrefix(CAPABILITY_PREFIX, next)),
            inputModalities: toModelModalities(withoutPrefix(INPUT_PREFIX, next)),
          })
        }
      />
    </Stack>
  );
};
