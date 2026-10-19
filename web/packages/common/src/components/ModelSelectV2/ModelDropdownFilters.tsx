// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  toModelSizeBucket,
  type ModelFilterControls,
  type ModelFilterOption,
  type ModelFilterValues,
} from '@nemo/common/src/api/models/modelFilters';
import { Flex, Select } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface ModelFilterSelectProps {
  name: keyof ModelFilterValues;
  label: string;
  value?: string;
  options: ModelFilterOption[];
  placeholder: string;
  disabled?: boolean;
  onChange: (value: string | undefined) => void;
}

const ModelFilterSelect: FC<ModelFilterSelectProps> = ({
  name,
  label,
  value,
  options,
  placeholder,
  disabled,
  onChange,
}) => (
  <Select
    className="min-w-0 flex-1"
    size="small"
    placeholder={placeholder}
    value={value ?? ''}
    dismissible
    disabled={disabled}
    onValueChange={(next) => onChange(next || undefined)}
    items={options.map((option) => ({ value: option.value, children: option.label }))}
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
  familyOptions,
  sizeOptions,
  providersLoading,
}) => (
  <Flex gap="density-sm" className="px-2 pb-2 w-full" data-testid="model-select-v2-filters">
    <ModelFilterSelect
      name="provider"
      label="Filter by provider"
      placeholder={providersLoading ? 'Loading...' : 'Provider'}
      value={values.provider}
      options={providerOptions}
      disabled={!values.provider && providerOptions.length === 0}
      onChange={(provider) => onChange({ ...values, provider })}
    />
    <ModelFilterSelect
      name="family"
      label="Filter by family"
      placeholder="Family"
      value={values.family}
      options={familyOptions}
      disabled={!values.family && familyOptions.length === 0}
      onChange={(family) => onChange({ ...values, family })}
    />
    <ModelFilterSelect
      name="size"
      label="Filter by size"
      placeholder="Size"
      value={values.size}
      options={sizeOptions}
      onChange={(size) => onChange({ ...values, size: toModelSizeBucket(size) })}
    />
  </Flex>
);
