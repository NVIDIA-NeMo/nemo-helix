// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PaginationQueryState } from '@nemo/common/src/utils/useQueryFromSearchParams';
import { Filter } from '@nemo/icons/icons';
import {
  Dropdown,
  type DropdownEntry,
  type DropdownProps,
  Flex,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { FilterTags } from '@studio/components/common/FilterToolbar/components/FilterTags';
import { SearchBar } from '@studio/components/common/SearchBar';
import { ComponentProps, useMemo } from 'react';

interface Props<K> {
  searchBarProps?: ComponentProps<typeof SearchBar>;
  disabled?: boolean;
  items: DropdownEntry[];
  countLabel?: React.ReactNode;
  onRemoveAllFilters: () => void;
  query: PaginationQueryState & K;
  setQuery: (newQuery: PaginationQueryState | K) => void;
  onItemCheckedChange?: DropdownProps['onItemCheckedChange'];
}

/**
 * A visual component for filtering data based on a search query or other filters.
 * @returns A search bar that filters the data based on the search query.
 */
export const FilterToolbar = <K,>({
  disabled,
  items,
  searchBarProps,
  countLabel,
  onRemoveAllFilters,
  query,
  setQuery,
  onItemCheckedChange,
}: Props<K>) => {
  const { onSubmit } = searchBarProps || {};

  const activeFilters = useMemo(
    () => items.filter((filter) => typeof filter !== 'string' && 'value' in filter && filter.value),
    [items]
  );

  return (
    <Stack gap="density-md">
      <Flex gap="density-md" align="center">
        <Dropdown onItemCheckedChange={onItemCheckedChange} items={items} disabled={disabled}>
          <Filter size="16px" /> Filter
        </Dropdown>

        <SearchBar
          label=""
          defaultValue={query.q}
          formClassName="flex-1"
          {...searchBarProps}
          disabled={disabled ?? searchBarProps?.disabled}
          onSubmit={(searchValue) => {
            setQuery({ q: searchValue });
            onSubmit?.(searchValue);
          }}
        />
        {countLabel && <Text>{countLabel}</Text>}
      </Flex>
      {activeFilters.length > 0 && (
        <FilterTags filters={activeFilters} onRemoveAllFilters={onRemoveAllFilters} />
      )}
    </Stack>
  );
};
