// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  toModelCapabilities,
  toModelModalities,
  toModelSizeBuckets,
  type ModelFilterControls,
  type ModelFilterOption,
} from '@nemo/common/src/api/models/modelFilters';
import { Flex, Select, type SelectEntry } from '@nvidia/foundations-react-core';
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
  className?: string;
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
  className = 'min-w-0',
  listSelected = false,
  disabled,
  onChange,
}) => (
  <Select
    className={className}
    size="small"
    multiple
    placeholder={placeholder}
    value={[...values]}
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
    <Flex gap="density-sm" className="px-2 pb-2 w-full" data-testid="model-select-v2-filters">
      <ModelFilterSelect
        name="providers"
        label="Filter by provider"
        className="min-w-0 w-[30%] shrink-0"
        placeholder={providersLoading ? 'Loading...' : 'Provider'}
        values={values.providers}
        options={providerOptions}
        disabled={!values.providers?.length && providerOptions.length === 0}
        onChange={(providers) => onChange({ ...values, providers })}
      />
      <ModelFilterSelect
        name="sizes"
        label="Filter by size"
        className="min-w-0 w-[24%] shrink-0"
        placeholder="Size"
        values={values.sizes}
        options={sizeOptions}
        onChange={(next) => onChange({ ...values, sizes: toModelSizeBuckets(next) })}
      />
      <ModelFilterSelect
        name="capabilities"
        label="Filter by capability"
        className="min-w-0 flex-1"
        placeholder="Capabilities"
        listSelected
        values={[
          ...(values.capabilities ?? []).map((value) => `${CAPABILITY_PREFIX}${value}`),
          ...(values.inputModalities ?? []).map((value) => `${INPUT_PREFIX}${value}`),
        ]}
        options={[...prefixedCapabilities, ...prefixedInputs]}
        items={[
          { kind: 'section', slotHeading: 'Features', items: toItems(prefixedCapabilities) },
          { kind: 'section', slotHeading: 'Input', items: toItems(prefixedInputs) },
        ]}
        onChange={(next) =>
          onChange({
            ...values,
            capabilities: toModelCapabilities(withoutPrefix(CAPABILITY_PREFIX, next)),
            inputModalities: toModelModalities(withoutPrefix(INPUT_PREFIX, next)),
          })
        }
      />
    </Flex>
  );
};
