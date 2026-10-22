// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import {
  DebouncedTextInput,
  type DebouncedTextInputProps,
} from '@nemo/common/src/components/DataView/internal/DebouncedTextInput';
import { MagnifyingGlass } from '@nvidia/react-gui-icons/icons';
import { useCallback, type JSX } from 'react';

export interface DataViewSearchBarProps extends Omit<
  Partial<DebouncedTextInputProps>,
  'placeholder'
> {
  /**
   * Provide a placeholder that informs the user what they're able to search for.
   * @defaultValue "MagnifyingGlass table"
   */
  placeholder?: string;
}

const MagnifyingGlassIcon = <MagnifyingGlass variant="line" />;

/**
 * A search bar for the data view. Should be rendered in `DataView.Toolbar`. Controls the
 * DataView's "global filter" — i.e. cross-column search.
 */
export function SearchBar(props: DataViewSearchBarProps): JSX.Element {
  const { state } = useInnerDataViewContext();
  const setSearchBar = state.searchBar.set;
  const resetPaginationToFirstPage = state.pagination.goToFirstPage;
  const handleChange = useCallback(
    (value: string) => {
      setSearchBar(value);
      resetPaginationToFirstPage();
    },
    [setSearchBar, resetPaginationToFirstPage]
  );
  return (
    <DebouncedTextInput
      dismissible
      onValueChange={handleChange}
      placeholder="MagnifyingGlass table"
      slotStart={MagnifyingGlassIcon}
      value={state.searchBar.state}
      {...props}
    />
  );
}
