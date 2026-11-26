// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetComboboxRow } from '@nemo/common/src/components/FilesetPicker/FilesetComboboxRow';
import { FilesetStorageIcon } from '@nemo/common/src/components/FilesetPicker/FilesetStorageIcon';
import type {
  FilesetPickerNoun,
  FilesetPickerPurpose,
} from '@nemo/common/src/components/FilesetPicker/types';
import { useFilesetSearch } from '@nemo/common/src/components/FilesetSearchableSelect/useFilesetSearch';
import { TextInputSpinner } from '@nemo/common/src/components/form/TextInputSpinner';
import { getEntityReference, getPartsFromReference } from '@nemo/common/src/namedEntity';
import { Add } from '@nemo/icons/icons';
import { useFilesRetrieveFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Combobox, type ComboboxEntry, Text } from '@nvidia/foundations-react-core';
import { type FC, useEffect, useMemo, useState } from 'react';
import { useDebounce } from 'use-debounce';

/** Never a fileset reference: those always contain a `/`. */
const NEW_FILESET_VALUE = '__new_fileset__';
const EMPTY_MESSAGE_VALUE = '__fileset_empty__';
const SEARCH_DEBOUNCE_MS = 300;

export interface FilesetComboboxProps {
  workspace: string;
  purpose?: FilesetPickerPurpose;
  value: string | null;
  onChange: (value: string | null, fileset?: FilesetOutput) => void;
  noun: FilesetPickerNoun;
  onRequestNew?: () => void;
  disabled?: boolean;
  status?: 'error' | 'success';
  onBlur?: () => void;
  /** Called with true when `value` names a fileset that no longer exists. */
  onMissingChange?: (missing: boolean) => void;
  id?: string;
  'aria-label'?: string;
}

export const FilesetCombobox: FC<FilesetComboboxProps> = ({
  workspace,
  purpose,
  value,
  onChange,
  noun,
  onRequestNew,
  disabled,
  status,
  onBlur,
  onMissingChange,
  id,
  'aria-label': ariaLabel,
}) => {
  const [inputText, setInputText] = useState('');
  const [debouncedInput] = useDebounce(inputText, SEARCH_DEBOUNCE_MS);
  const [picked, setPicked] = useState<FilesetOutput | null>(null);

  const { filesets, search, setSearch, loadMore, isLoading, isLoadingMore, isError } =
    useFilesetSearch({ workspace, purpose });

  const loaded = value ? filesets.find((fileset) => getEntityReference(fileset) === value) : null;
  const pickedMatches = picked && getEntityReference(picked) === value ? picked : null;
  const { workspace: valueWorkspace, name: valueName } = getPartsFromReference(value ?? '');
  const { data: retrieved, isError: isMissing } = useFilesRetrieveFileset(
    valueWorkspace,
    valueName,
    { query: { enabled: Boolean(value && valueName && !isLoading && !loaded && !pickedMatches) } }
  );
  const selected = loaded ?? pickedMatches ?? (retrieved && value ? retrieved : null);
  const missing = Boolean(value && !selected && isMissing);

  useEffect(() => {
    onMissingChange?.(missing);
  }, [missing, onMissingChange]);

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
        : missing && value
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
    missing,
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
