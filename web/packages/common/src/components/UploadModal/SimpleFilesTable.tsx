// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { LeftOverflow } from '@nemo/common/src/components/LeftOverflow';
import { useUploadModalContext } from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import { useInlinePickerSlot } from '@nemo/common/src/components/UploadModal/InlinePickerSlot';
import { UploadFile } from '@nemo/common/src/components/UploadModal/types';
import { formatFileSize, isFileTypeAllowed } from '@nemo/common/src/components/UploadModal/utils';
import { IconError } from '@nemo/icons/icons';
import {
  Button,
  Checkbox,
  Flex,
  RadioGroupInput,
  RadioGroupItem,
  RadioGroupRoot,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { type ComponentProps, useCallback, useEffect, useMemo, useRef } from 'react';

type FileRow = {
  id: string;
  name: string;
  size: number;
  isDisabled: boolean;
  uploadFile: UploadFile;
};

export const SimpleFilesTable = () => {
  const [state, dispatch] = useUploadModalContext();
  const { trailingButton } = useInlinePickerSlot();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const {
    files,
    selectedFiles,
    errors,
    acceptableFileTypes,
    allowMultipleFileSelection,
    invalidFileMode,
  } = state;

  // An empty `acceptableFileTypes` accepts every file, so no separate "is restricted" flag is needed.
  const visibleFiles = useMemo(
    () =>
      invalidFileMode === 'hide'
        ? files.filter((file) => isFileTypeAllowed(file, acceptableFileTypes))
        : files,
    [files, acceptableFileTypes, invalidFileMode]
  );

  const hasValidSelection = selectedFiles.some((file) =>
    isFileTypeAllowed(file, acceptableFileTypes)
  );
  const disabledFilesMessage =
    invalidFileMode === 'disable' &&
    !hasValidSelection &&
    visibleFiles.some((file) => !isFileTypeAllowed(file, acceptableFileTypes))
      ? `Only ${acceptableFileTypes.join(', ')} files can be selected. Upload a supported file or choose a different fileset.`
      : null;

  const fileRows = useMemo<FileRow[]>(
    () =>
      visibleFiles.map((uploadFile) => {
        const isDisabled =
          invalidFileMode === 'disable' && !isFileTypeAllowed(uploadFile, acceptableFileTypes);
        const name = uploadFile.type === 'existing' ? uploadFile.file.path : uploadFile.file.name;
        const size = uploadFile.type === 'existing' ? uploadFile.file.size : uploadFile.file.size;
        return { id: uploadFile.id, name, size, isDisabled, uploadFile };
      }),
    [visibleFiles, invalidFileMode, acceptableFileTypes]
  );

  const dataViewState = DataView.useDataViewState({ pagination: { pageSize: 10 } });

  const searchTerm = dataViewState.searchBar.state.trim().toLowerCase();
  const matchingRows = useMemo(
    () =>
      searchTerm ? fileRows.filter((row) => row.name.toLowerCase().includes(searchTerm)) : fileRows,
    [fileRows, searchTerm]
  );
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const safePageIndex = Math.min(
    pageIndex,
    Math.max(0, Math.ceil(matchingRows.length / pageSize) - 1)
  );
  const pageRows = useMemo(
    () => matchingRows.slice(safePageIndex * pageSize, safePageIndex * pageSize + pageSize),
    [matchingRows, safePageIndex, pageSize]
  );

  const setPagination = dataViewState.pagination.set;
  useEffect(() => {
    if (pageIndex !== safePageIndex) {
      setPagination((prev) => ({ ...prev, pageIndex: safePageIndex }));
    }
  }, [pageIndex, safePageIndex, setPagination]);

  const makeColumns = useCallback<ComponentProps<typeof DataView.Root<FileRow>>['makeColumns']>(
    (col) => [
      col.display({
        id: 'row-selection',
        header: () => null,
        size: 48,
        maxSize: 48,
        minSize: 48,
        meta: { alignment: 'center' as const, _isPrebuiltColumn: true, _isSizeInitialized: true },
        cell: ({ row }) =>
          allowMultipleFileSelection ? (
            <Checkbox
              checked={selectedFiles.some((f) => f.id === row.original.id)}
              onCheckedChange={() =>
                dispatch({ type: 'TOGGLE_FILE_SELECTION', payload: row.original.uploadFile })
              }
              disabled={row.original.isDisabled}
              attributes={{ CheckboxInput: { 'aria-label': row.original.name } }}
            />
          ) : (
            <RadioGroupItem aria-label={row.original.name}>
              <RadioGroupInput value={row.original.id} disabled={row.original.isDisabled} />
            </RadioGroupItem>
          ),
      }),
      col.accessor('name', {
        header: 'Name',
        cell: (ctx) => <LeftOverflow>{ctx.getValue()}</LeftOverflow>,
      }),
      col.accessor('size', {
        header: 'Size',
        size: 120,
        maxSize: 120,
        minSize: 120,
        meta: { _isSizeInitialized: true },
        cell: (ctx) => formatFileSize(ctx.getValue()),
      }),
    ],
    [allowMultipleFileSelection, selectedFiles, dispatch]
  );

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newFiles = event.target.files;
    if (newFiles) {
      dispatch({
        type: 'SET_FILES',
        payload: Array.from(newFiles).map((file) => ({ id: file.name, type: 'new', file })),
      });
    }
  };

  return (
    <Stack className="min-h-0 flex-1 w-full" gap="density-md">
      <RadioGroupRoot
        className="min-h-0 max-h-[60dvh] flex"
        name="simple-files-table"
        value={selectedFiles[0]?.id ?? ''}
        onValueChange={(id) => {
          const file = fileRows.find((r) => r.id === id);
          if (file) dispatch({ type: 'TOGGLE_FILE_SELECTION', payload: file.uploadFile });
        }}
      >
        <StudioDataView<FileRow>
          dataViewState={dataViewState}
          makeColumns={makeColumns}
          searchField="name"
          maxTwoLines={false}
          onRowClick={(row) => {
            if (!row.isDisabled) {
              dispatch({ type: 'TOGGLE_FILE_SELECTION', payload: row.uploadFile });
            }
          }}
          attributes={{
            DataViewRoot: {
              data: pageRows,
              totalCount: matchingRows.length,
              reactTableOptions: { getRowId: (row) => row.id },
            },
            DataViewSearchBar: { placeholder: 'Search files…' },
            DataViewPagination: { pageSizeOptions: [10, 25, 50] },
          }}
        />
      </RadioGroupRoot>
      {disabledFilesMessage ? (
        <Flex gap="density-sm" align="center">
          <IconError className="text-feedback-warning shrink-0" />
          <Text kind="label/regular/sm" className="text-feedback-warning">
            {disabledFilesMessage}
          </Text>
        </Flex>
      ) : null}
      {errors.file && (
        <Text kind="label/regular/sm" className="text-feedback-danger">
          {errors.file}
        </Text>
      )}
      {trailingButton ? (
        <Flex justify="between" align="center">
          <Button
            kind="tertiary"
            onClick={() => {
              fileInputRef.current?.click();
            }}
          >
            Upload More Files
          </Button>
          {trailingButton}
        </Flex>
      ) : (
        <Button
          kind="tertiary"
          onClick={() => {
            fileInputRef.current?.click();
          }}
        >
          Upload More Files
        </Button>
      )}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        tabIndex={-1}
        onChange={handleFileChange}
        accept={acceptableFileTypes.join(',')}
        className="sr-only"
        aria-label="Upload more files"
      />
    </Stack>
  );
};
