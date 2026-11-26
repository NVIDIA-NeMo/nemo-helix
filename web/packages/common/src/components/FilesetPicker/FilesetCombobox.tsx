// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import {
  getFilesetQueryKey,
  type ListFilesets,
  type RetrieveFileset,
} from '@nemo/common/src/api/filesets/filesetPages';
import { useFilesetSearchCore } from '@nemo/common/src/api/filesets/useFilesetSearchCore';
import { FilesetComboboxRow } from '@nemo/common/src/components/FilesetPicker/FilesetComboboxRow';
import { FilesetStorageIcon } from '@nemo/common/src/components/FilesetPicker/FilesetStorageIcon';
import type {
  FilesetPickerNoun,
  FilesetPickerPurpose,
  FilesetSelectionError,
} from '@nemo/common/src/components/FilesetPicker/types';
import { TextInputSpinner } from '@nemo/common/src/components/form/TextInputSpinner';
import { getEntityReference, getPartsFromReference } from '@nemo/common/src/namedEntity';
import { Add } from '@nemo/icons/icons';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Combobox, type ComboboxEntry, Text } from '@nvidia/foundations-react-core';
import { useQuery } from '@tanstack/react-query';
import { type FC, useEffect, useMemo, useState } from 'react';
import { useDebounce } from 'use-debounce';

/** Never a fileset reference: those always contain a `/`. */
const NEW_FILESET_VALUE = '__new_fileset__';
const EMPTY_MESSAGE_VALUE = '__fileset_empty__';
const SEARCH_DEBOUNCE_MS = 300;

export interface FilesetComboboxProps {
  workspace: string;
  listFilesets: ListFilesets;
  retrieveFileset: RetrieveFileset;
  purpose?: FilesetPickerPurpose;
  value: string | null;
  onChange: (value: string | null, fileset?: FilesetOutput) => void;
  noun: FilesetPickerNoun;
  onRequestNew?: () => void;
  disabled?: boolean;
  status?: 'error' | 'success';
  onBlur?: () => void;
  /** Reports why `value` couldn't be resolved: deleted (`missing`) or the lookup failed. */
  onSelectionErrorChange?: (error: FilesetSelectionError | null) => void;
  id?: string;
  'aria-label'?: string;
}

export const FilesetCombobox: FC<FilesetComboboxProps> = ({
  workspace,
  listFilesets,
  retrieveFileset,
  purpose,
  value,
  onChange,
  noun,
  onRequestNew,
  disabled,
  status,
  onBlur,
  onSelectionErrorChange,
  id,
  'aria-label': ariaLabel,
}) => {
  const [inputText, setInputText] = useState('');
  const [debouncedInput] = useDebounce(inputText, SEARCH_DEBOUNCE_MS);
  const [picked, setPicked] = useState<FilesetOutput | null>(null);

  const { filesets, search, setSearch, loadMore, isLoading, isLoadingMore, isError } =
    useFilesetSearchCore({ workspace, listFilesets, purpose });

  const loaded = value ? filesets.find((fileset) => getEntityReference(fileset) === value) : null;
  const pickedMatches = picked && getEntityReference(picked) === value ? picked : null;
  const { workspace: valueWorkspace, name: valueName } = getPartsFromReference(value ?? '');
  // Remember the selection once the results include it, so it can still be shown while a
  // search hides it.
  if (loaded && loaded !== picked) setPicked(loaded);

  // Look the selection up when the results don't contain it. A search hiding an already-known
  // selection is not a reason to refetch; a refreshed list that dropped it is.
  const { data: retrieved, error: lookupError } = useQuery({
    queryKey: getFilesetQueryKey(valueWorkspace, valueName),
    queryFn: ({ signal }) => retrieveFileset(valueWorkspace, valueName, signal),
    enabled: Boolean(value && valueName && !isLoading && !loaded && !(search && pickedMatches)),
  });
  // React Query keeps the last good result after a failed refetch; a 404 must still win.
  const notFound = !loaded && isNotFoundError(lookupError);
  const selected = loaded ?? (notFound ? null : ((value ? retrieved : null) ?? pickedMatches));
  const selectionError: FilesetSelectionError | null =
    value && !selected && lookupError
      ? isNotFoundError(lookupError)
        ? 'missing'
        : 'unavailable'
      : null;

  useEffect(() => {
    onSelectionErrorChange?.(selectionError);
  }, [selectionError, onSelectionErrorChange]);

  // Picking an option writes its name into the input. Treat that as "no search" so reopening
  // the list shows every fileset rather than just the one already chosen.
  useEffect(() => {
    const term = debouncedInput.trim();
    setSearch(selected && term === selected.name ? '' : term);
  }, [debouncedInput, selected, setSearch]);

  const showPurpose = Array.isArray(purpose) && purpose.length > 1;

  const emptyMessage = isError
    ? `Couldn't load ${noun.many}`
    : isLoading
      ? `Loading ${noun.many}…`
      : search
        ? `No ${noun.many} match "${search}" — try a different name`
        : `No ${noun.many} in this workspace yet`;

  const items = useMemo<ComboboxEntry[]>(() => {
    const results: ComboboxEntry[] = filesets.map((fileset) => ({
      value: getEntityReference(fileset),
      filterValue: fileset.name,
      children: <FilesetComboboxRow fileset={fileset} showPurpose={showPurpose} />,
    }));

    // KUI can only show (and restore on blur) the input label of a value that is an item, so the
    // selection stays in the list even when the results don't include it.
    const selectionOutsideResults: ComboboxEntry | null =
      selected && !filesets.some((f) => getEntityReference(f) === getEntityReference(selected))
        ? {
            value: getEntityReference(selected),
            filterValue: selected.name,
            children: <FilesetComboboxRow fileset={selected} showPurpose={showPurpose} />,
          }
        : selectionError && value
          ? {
              value,
              filterValue: valueName,
              disabled: true,
              children: (
                <Text kind="body/regular/sm" className="text-secondary">
                  {valueName}
                </Text>
              ),
            }
          : null;

    const newRow: ComboboxEntry[] = onRequestNew
      ? [
          {
            value: NEW_FILESET_VALUE,
            filterValue: null,
            children: (
              <span className="flex w-full items-center gap-2">
                <Add size="16" className="shrink-0 text-primary" aria-hidden="true" />
                <Text kind="label/semibold/sm" className="text-primary">
                  New {noun.one}
                </Text>
              </span>
            ),
          },
        ]
      : [];

    // While searching, a selection outside the results gets its own heading so it does not read
    // as a match. Without a search it simply leads the list.
    const selection: ComboboxEntry[] = !selectionOutsideResults
      ? []
      : search
        ? [{ kind: 'section', slotHeading: 'Selected', items: [selectionOutsideResults] }]
        : [selectionOutsideResults];

    // Any extra row keeps the listbox non-empty, which suppresses KUI's own empty state.
    const emptyRow: ComboboxEntry[] =
      results.length === 0 && !isLoading && newRow.length + selection.length > 0
        ? [
            {
              value: EMPTY_MESSAGE_VALUE,
              filterValue: null,
              disabled: true,
              children: (
                <Text kind="body/regular/sm" className="text-secondary">
                  {emptyMessage}
                </Text>
              ),
            },
          ]
        : [];

    return search
      ? [...newRow, ...emptyRow, ...results, ...selection]
      : [...newRow, ...emptyRow, ...selection, ...results];
  }, [
    filesets,
    selected,
    search,
    selectionError,
    value,
    valueName,
    showPurpose,
    onRequestNew,
    noun.one,
    isLoading,
    emptyMessage,
  ]);

  const handleValueChange = (next: string) => {
    if (next === NEW_FILESET_VALUE) {
      onRequestNew?.();
      return;
    }
    if (next === EMPTY_MESSAGE_VALUE) return;
    if (!next) {
      setPicked(null);
      onChange(null);
      return;
    }
    const fileset =
      filesets.find((f) => getEntityReference(f) === next) ??
      (selected && getEntityReference(selected) === next ? selected : undefined);
    setPicked(fileset ?? null);
    onChange(next, fileset);
  };

  return (
    <Combobox
      id={id}
      aria-label={ariaLabel}
      items={items}
      value={value ?? ''}
      onValueChange={handleValueChange}
      onInputValueChange={setInputText}
      attributes={{ ComboboxTrigger: { onBlur } }}
      filterMatchFn="disable"
      onScrollToBottom={() => void loadMore()}
      emptyStateMessage={emptyMessage}
      placeholder={`Search ${noun.many}…`}
      dismissible
      disabled={disabled}
      status={status}
      slotStart={selected ? <FilesetStorageIcon fileset={selected} /> : undefined}
      slotEnd={isLoading || isLoadingMore ? <TextInputSpinner /> : undefined}
    />
  );
};
