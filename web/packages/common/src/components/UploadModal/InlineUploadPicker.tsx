// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { UploadModalProvider } from '@nemo/common/src/components/UploadModal/Context/UploadModalProvider';
import { useUploadModalContext } from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import { uploadModalInitialState } from '@nemo/common/src/components/UploadModal/Context/useUploadModalReducer';
import { InlinePickerSlotProvider } from '@nemo/common/src/components/UploadModal/InlinePickerSlot';
import { SubmitUploadType, UploadModalProps } from '@nemo/common/src/components/UploadModal/types';
import { UploadPickerBody } from '@nemo/common/src/components/UploadModal/UploadPickerBody';
import { useUploadSubmit } from '@nemo/common/src/components/UploadModal/useUploadSubmit';
import { Flex, Stack } from '@nvidia/foundations-react-core';
import { FC, MouseEvent, useCallback, useEffect, useMemo, useRef } from 'react';

type InlineUploadPickerProps = Pick<
  UploadModalProps,
  | 'workspace'
  | 'includeDataset'
  | 'includeTabs'
  | 'allowMultipleFileSelection'
  | 'acceptableFileTypes'
  | 'acceptableFileSize'
  | 'invalidFileMode'
  | 'allowNewDataset'
  | 'filesetPurpose'
  | 'datasetLabel'
  | 'autoSelectFirstAcceptable'
  | 'showUpdatedAt'
> & {
  /** Called once the picked / uploaded file is committed. */
  onSubmit: (data: SubmitUploadType) => void;
  /** Label of the inline "Add" button. */
  addButtonText?: string;
  /** When true, skip the explicit "Add" button: as soon as the user picks a
   *  valid file (radio selection on an existing fileset), commit it to the
   *  parent form. Intended for picker surfaces where uploading new files
   *  isn't the primary flow — extra clicks just slow the user down. */
  autoCommit?: boolean;
};

interface InnerProps {
  workspace: string;
  includeDataset: boolean;
  includeTabs: boolean;
  onSubmit: (data: SubmitUploadType) => void;
  addButtonText: string;
  autoCommit: boolean;
}

const InlineUploadPickerContent: FC<InnerProps> = ({
  workspace,
  includeDataset,
  includeTabs,
  onSubmit,
  addButtonText,
  autoCommit,
}) => {
  const [state] = useUploadModalContext();
  const { submit, isSubmitting } = useUploadSubmit({
    workspace,
    includeDataset,
    includeTabs,
    onSubmit,
  });

  // Track which selection we've already auto-committed so a re-render doesn't
  // re-fire the upload for the same picked file. Auto-commit is restricted to
  // the existing-dataset / existing-file path: new-dataset uploads are
  // side-effecting (filesCreateFileset + filesUploadFile) and would race with
  // the user editing the dataset name field.
  const committedRef = useRef<string | null>(null);
  const selectionKey = useMemo(() => {
    if (!autoCommit) return null;
    if (state.dataset?.type !== 'existing') return '';
    const hasNewFiles = state.selectedFiles.some((f) => f.type === 'new');
    if (hasNewFiles) return '';
    const datasetKey = `existing:${state.dataset.dataset.workspace}/${state.dataset.dataset.name}`;
    const fileKeys = state.selectedFiles.map((f) => f.id).join('|');
    if (!fileKeys) return '';
    return `${datasetKey}#${fileKeys}`;
  }, [autoCommit, state.dataset, state.selectedFiles]);

  useEffect(() => {
    if (!autoCommit) return;
    if (!selectionKey) return;
    if (committedRef.current === selectionKey) return;
    // Skip while a prior auto-commit is still in flight; on rapid selection
    // changes the next state update will re-trigger this effect after the
    // in-flight ``submit`` resolves.
    if (isSubmitting) return;
    committedRef.current = selectionKey;
    void submit();
  }, [autoCommit, selectionKey, isSubmitting, submit]);

  const handleAdd = useCallback(
    async (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      await submit();
    },
    [submit]
  );

  const addButton = useMemo(
    () =>
      autoCommit ? null : (
        <LoadingButton type="button" color="brand" loading={isSubmitting} onClick={handleAdd}>
          {addButtonText}
        </LoadingButton>
      ),
    [autoCommit, isSubmitting, handleAdd, addButtonText]
  );

  const slotValue = useMemo(() => ({ trailingButton: addButton }), [addButton]);

  if (!workspace) {
    console.error('InlineUploadPicker: workspace is required');
    return null;
  }

  // When SimpleFilesTable is rendering (i.e. the user has files staged), the
  // Add button moves into its bottom row alongside "Upload More Files" via
  // {@link InlinePickerSlotContext}. Otherwise it sits at the bottom of the
  // picker as a fallback.
  const filesTableVisible = state.files.length > 0;

  return (
    <InlinePickerSlotProvider value={slotValue}>
      <Stack gap="density-md" className="w-full">
        <UploadPickerBody
          workspace={workspace}
          includeDataset={includeDataset}
          includeTabs={includeTabs}
        />
        {addButton && !filesTableVisible ? <Flex justify="end">{addButton}</Flex> : null}
      </Stack>
    </InlinePickerSlotProvider>
  );
};

/**
 * Inline variant of {@link UploadModal}: same dataset/file picker UI, but
 * rendered in-place rather than behind a "Select File" button + secondary
 * modal layer. Use inside a parent form/modal when you want the picker to
 * live alongside the rest of the form's inputs.
 */
export const InlineUploadPicker: FC<InlineUploadPickerProps> = ({
  workspace,
  includeDataset = false,
  includeTabs = false,
  allowMultipleFileSelection,
  acceptableFileTypes,
  acceptableFileSize,
  invalidFileMode,
  allowNewDataset,
  filesetPurpose,
  datasetLabel,
  autoSelectFirstAcceptable,
  showUpdatedAt,
  onSubmit,
  addButtonText = 'Add file',
  autoCommit = false,
}) => {
  // In ``autoCommit`` mode the "Create new dataset" path would race with the
  // user editing the dataset name (each keystroke would re-fire the upload),
  // so suppress that option unless the consumer explicitly opts in.
  const effectiveAllowNewDataset = allowNewDataset ?? !autoCommit;
  // ``useReducer`` only reads ``initialState`` on mount, so this only matters
  // for first render — but memoizing also keeps the object identity stable in
  // case React StrictMode double-invokes the reducer init.
  const initialState = useMemo(
    () => ({
      ...uploadModalInitialState,
      allowMultipleFileSelection:
        allowMultipleFileSelection ?? uploadModalInitialState.allowMultipleFileSelection,
      acceptableFileTypes: acceptableFileTypes ?? uploadModalInitialState.acceptableFileTypes,
      acceptableFileSize: acceptableFileSize ?? uploadModalInitialState.acceptableFileSize,
      invalidFileMode: invalidFileMode ?? uploadModalInitialState.invalidFileMode,
      allowNewDataset: effectiveAllowNewDataset,
      filesetPurpose: filesetPurpose ?? uploadModalInitialState.filesetPurpose,
      datasetLabel: datasetLabel ?? uploadModalInitialState.datasetLabel,
      autoSelectFirstAcceptable:
        autoSelectFirstAcceptable ?? uploadModalInitialState.autoSelectFirstAcceptable,
      showUpdatedAt: showUpdatedAt ?? uploadModalInitialState.showUpdatedAt,
    }),
    [
      allowMultipleFileSelection,
      acceptableFileTypes,
      acceptableFileSize,
      invalidFileMode,
      effectiveAllowNewDataset,
      filesetPurpose,
      datasetLabel,
      autoSelectFirstAcceptable,
      showUpdatedAt,
    ]
  );
  return (
    <UploadModalProvider initialState={initialState}>
      <InlineUploadPickerContent
        workspace={workspace}
        includeDataset={includeDataset}
        includeTabs={includeTabs}
        onSubmit={onSubmit}
        addButtonText={addButtonText}
        autoCommit={autoCommit}
      />
    </UploadModalProvider>
  );
};
