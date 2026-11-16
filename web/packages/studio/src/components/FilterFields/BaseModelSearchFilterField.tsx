/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useBaseModels } from '@nemo/common/src/api/entity-store/useBaseModels';
import { ControlledSearchableSelect } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import { ModelEntitySortField } from '@nemo/sdk/generated/platform/schema/ModelEntitySortField';
import { SearchBaseModels } from '@studio/components/FilterFields/SearchBaseModels';
import { type Dispatch, type FC, type SetStateAction, useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';

export interface BaseModelSearchFilterFieldProps {
  workspace: string;
  /** Selected base model names (single string or array). */
  value: string | string[] | undefined;
  /** Called when selection changes. */
  onValueChange: (models: string[]) => void;
  /** Single-select mode: searchable select with infinite query. Multi-select uses combobox. */
  singleSelect?: boolean;
  /** data-testid for the single-select trigger (when singleSelect is true). */
  dataTestId?: string;
}

/** Normalizes value to string[] for SearchBaseModels. */
function normalizeToArray(value: string | string[] | undefined): string[] {
  if (value === undefined) return [];
  return Array.isArray(value) ? value : [value];
}

/** Single-select base model field: infinite query + ControlledSearchableSelect. */
const BaseModelSearchFilterFieldSingle: FC<
  Pick<BaseModelSearchFilterFieldProps, 'workspace' | 'value' | 'onValueChange' | 'dataTestId'>
> = ({ workspace, value, onValueChange, dataTestId }) => {
  const [searchInput, setSearchInput] = useState('');

  const selectedBaseModel =
    typeof value === 'string' ? value : Array.isArray(value) ? (value[0] ?? '') : '';

  const { models, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useBaseModels({
    workspace,
    filter: searchInput.trim() ? { name: { $like: searchInput.trim() } } : undefined,
    sort: ModelEntitySortField.name,
  });

  const form = useForm<{ base_model: string }>({
    defaultValues: { base_model: selectedBaseModel },
  });

  useEffect(() => {
    form.setValue('base_model', selectedBaseModel);
  }, [selectedBaseModel, form]);

  const options = useMemo(
    () =>
      models.map((model) => ({
        value: `${model.workspace}/${model.name ?? ''}`,
        label: model.name ?? '',
      })),
    [models]
  );

  const handleChange = (newValue: string) => {
    onValueChange(newValue ? [newValue] : []);
  };

  return (
    <ControlledSearchableSelect
      useControllerProps={{ control: form.control, name: 'base_model' }}
      name="base_model"
      data-testid={dataTestId ?? 'base-model-filter'}
      triggerPlaceholder="Select a base model"
      searchPlaceholder="Search base models..."
      options={options}
      onSearchChange={setSearchInput}
      onLoadMore={async () => {
        await fetchNextPage();
      }}
      hasMore={hasNextPage ?? false}
      isLoadingMore={isFetchingNextPage ?? false}
      isLoading={isLoading ?? false}
      onChange={handleChange}
      emptyMessage="No base models found"
      doneLoadingMessage="All base models loaded"
    />
  );
};

export const BaseModelSearchFilterField: FC<BaseModelSearchFilterFieldProps> = ({
  workspace,
  value,
  onValueChange,
  singleSelect = false,
  dataTestId,
}) => {
  if (singleSelect) {
    return (
      <BaseModelSearchFilterFieldSingle
        workspace={workspace}
        value={value}
        onValueChange={onValueChange}
        dataTestId={dataTestId}
      />
    );
  }

  const selectedModels = normalizeToArray(value);
  const setSelectedModels: Dispatch<SetStateAction<string[]>> = (action) => {
    const next = typeof action === 'function' ? action(selectedModels) : action;
    onValueChange(next);
  };
  return (
    <SearchBaseModels
      workspace={workspace}
      selectedModels={selectedModels}
      setSelectedModels={setSelectedModels}
    />
  );
};
