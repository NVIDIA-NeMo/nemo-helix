// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  toModelCapability,
  toModelModality,
  toModelSizeBucket,
  type ModelFilterControls,
  type ModelFilterOption,
  type ModelFilterValues,
} from '@nemo/common/src/api/models/modelFilters';
import { Grid, Select } from '@nvidia/foundations-react-core';
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
    className="min-w-0"
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
  capabilityOptions,
  inputModalityOptions,
  sizeOptions,
  providersLoading,
}) => (
  <Grid
    cols={2}
    gap="density-sm"
    className="px-2 pb-2 w-full"
    data-testid="model-select-v2-filters"
  >
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
      name="capability"
      label="Filter by capability"
      placeholder="Capability"
      value={values.capability}
      options={capabilityOptions}
      onChange={(capability) => onChange({ ...values, capability: toModelCapability(capability) })}
    />
    <ModelFilterSelect
      name="inputModality"
      label="Filter by input"
      placeholder="Input"
      value={values.inputModality}
      options={inputModalityOptions}
      onChange={(modality) => onChange({ ...values, inputModality: toModelModality(modality) })}
    />
    <ModelFilterSelect
      name="size"
      label="Filter by size"
      placeholder="Size"
      value={values.size}
      options={sizeOptions}
      onChange={(size) => onChange({ ...values, size: toModelSizeBucket(size) })}
    />
  </Grid>
);
