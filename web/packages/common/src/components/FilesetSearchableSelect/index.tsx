// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFilesetSearch } from '@nemo/common/src/components/FilesetSearchableSelect/useFilesetSearch';
import {
  ControlledSearchableSelect,
  type SelectItemOption,
} from '@nemo/common/src/components/form/ControlledSearchableSelect';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import type { FilesetOutput, FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { type ReactElement, type ReactNode, useCallback, useMemo } from 'react';
import { type FieldValues, type UseControllerProps } from 'react-hook-form';

export interface FilesetSearchableSelectFormFieldProps {
  slotLabel?: ReactNode;
  slotInfo?: ReactNode;
  slotError?: string;
}

export interface FilesetSearchableSelectProps<T extends FieldValues> {
  workspace: string;
  queryEnabled?: boolean;
  useControllerProps: UseControllerProps<T>;
  formFieldProps: FilesetSearchableSelectFormFieldProps;
  triggerPlaceholder?: string;
  /** Restrict to one fileset `purpose`. Omit to list every purpose. */
  purpose?: FilesetPurpose;
  /** Restrict to filesets carrying this exact `description`. */
  description?: string;
  /** Options rendered above the fileset list (e.g. a "New Dataset" entry). */
  leadingOptions?: SelectItemOption[];
  groupLabels?: Record<string, string>;
  /** Build the option row for a fileset. Defaults to its `workspace/name` reference. */
  renderOption?: (fileset: FilesetOutput) => SelectItemOption;
  /** Fired with the picked option value, alongside the form field update. The matching
   *  fileset is resolved from the loaded pages, and is undefined for `leadingOptions`. */
  onChange?: (value: string, fileset?: FilesetOutput) => void;
  disabled?: boolean;
}

const defaultRenderOption = (fileset: FilesetOutput): SelectItemOption => {
  const ref = getEntityReference(fileset);
  return { value: ref, label: ref };
};

/**
 * A fileset picker with server-side search and pagination.
 *
 * Prefer this over a plain `Select` fed by a single `filesListFilesets` page: that shape
 * caps out at the API's 100-item page and gives the user no way to reach the rest.
 */
export function FilesetSearchableSelect<T extends FieldValues>({
  workspace,
  queryEnabled = true,
  useControllerProps,
  formFieldProps,
  triggerPlaceholder = 'Select a fileset',
  purpose,
  description,
  leadingOptions,
  groupLabels,
  renderOption = defaultRenderOption,
  onChange,
  disabled,
}: FilesetSearchableSelectProps<T>): ReactElement {
  const { filesets, setSearch, loadMore, hasMore, isLoading, isLoadingMore, isError } =
    useFilesetSearch({
      workspace,
      purpose,
      description,
      enabled: queryEnabled,
    });

  const filesetOptions = useMemo(
    () => filesets.map((fileset) => ({ fileset, option: renderOption(fileset) })),
    [filesets, renderOption]
  );

  const options = useMemo<SelectItemOption[]>(
    () => [...(leadingOptions ?? []), ...filesetOptions.map(({ option }) => option)],
    [filesetOptions, leadingOptions]
  );

  const handleChange = useCallback(
    (value: string) => {
      onChange?.(value, filesetOptions.find(({ option }) => option.value === value)?.fileset);
    },
    [onChange, filesetOptions]
  );

  return (
    <ControlledSearchableSelect
      useControllerProps={useControllerProps}
      options={options}
      groupLabels={groupLabels}
      onChange={handleChange}
      disabled={disabled}
      onSearchChange={setSearch}
      onLoadMore={loadMore}
      hasMore={hasMore}
      isLoading={isLoading}
      isLoadingMore={isLoadingMore}
      searchPlaceholder="Search filesets..."
      emptyMessage={isError ? 'Failed to load filesets' : 'No filesets found'}
      triggerPlaceholder={triggerPlaceholder}
      formFieldProps={formFieldProps}
    />
  );
}
