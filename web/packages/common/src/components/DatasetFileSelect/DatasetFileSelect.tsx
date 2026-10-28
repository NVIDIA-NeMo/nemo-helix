// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DatasetFileSelectButton } from '@nemo/common/src/components/DatasetFileSelect/DatasetFileSelectButton';
import { useFilePreview } from '@nemo/common/src/components/DatasetFileSelect/hooks/useFilePreview';
import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { FileContentPreview } from '@nemo/common/src/components/FileContentPreview';
import { FileList, FileListItem } from '@nemo/common/src/components/FileList';
import { UploadModal } from '@nemo/common/src/components/UploadModal/index';
import { InlineUploadPicker } from '@nemo/common/src/components/UploadModal/InlineUploadPicker';
import type { SubmitUploadType } from '@nemo/common/src/components/UploadModal/types';
import { getFileExtension, isAcceptedFileType } from '@nemo/common/src/utils/file';
import { FolderOpen } from '@nemo/icons/icons';
import type { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { SidePanel, Stack, Text } from '@nvidia/foundations-react-core';
import { FC, useEffect, useMemo, useRef, useState } from 'react';

export type AcceptedFileType = '.json' | '.jsonl' | '.csv' | '.parquet' | '.yml' | '.yaml';

interface DatasetFileSelectProps {
  /** The selected file(s). This is a controlled component - parent owns the state. */
  value?: FileListItem | FileListItem[] | null;
  showSplitInformation?: boolean;
  holdoutSplitPercentage?: number;
  acceptedFileTypes?: AcceptedFileType[];
  /** How to render existing files whose extension isn't in
   *  ``acceptedFileTypes``. ``'show'`` (default) renders everything;
   *  ``'hide'`` filters them out; ``'disable'`` renders them but blocks
   *  selection. */
  invalidFileMode?: 'show' | 'hide' | 'disable';
  /** Called when files change. Required for the component to function. */
  onChange: (files: FileListItem[]) => void;
  onError?: (error: { message: string; filepath?: string }) => void;
  onClearError?: () => void;
  error?: string;
  workspace: string;
  errorText?: string;
  /** If true, allows multiple file selection. */
  allowMultiple?: boolean;
  /** If true, shows both dataset and file upload options in tabs. */
  includeTabs?: boolean;
  /** If true, shows dataset selection UI. If false, only shows file upload. */
  includeDataset?: boolean;
  /** The label for the file list. */
  listLabel?: string;
  /** When true, renders the picker UI inline (no "Select File" button + no
   *  secondary modal) so it lives alongside the rest of the parent form. */
  inline?: boolean;
  /** Inline-only: skip the "Add" button and commit directly when the user
   *  selects a file. Also hides the post-commit file list since the parent
   *  form already reflects the selection. */
  autoCommit?: boolean;
  /** Fileset ``purpose`` the picker lists. Defaults to ``'dataset'``. */
  filesetPurpose?: FilesetPurpose;
  /** Let one upload carry several files while the selection stays a single file. */
  allowMultipleUpload?: boolean;
  /** Label for the fileset picker. Defaults to ``'Dataset'``. */
  datasetLabel?: string;
  /** Auto-select the first root-level accepted file on fileset selection. */
  autoSelectFirstAcceptable?: boolean;
  showUpdatedAt?: boolean;
}

/**
 * Controlled dataset file select component with button, modal, and file list.
 * Supports both file uploads and dataset selection.
 *
 * This is a controlled component - the parent must provide `value` and `onChange`.
 *
 * For more granular control, use the individual components:
 * - DatasetFileSelectButton
 * - FileList
 * - useFilePreview (hook for preview functionality)
 */
export const DatasetFileSelect: FC<DatasetFileSelectProps> = ({
  value,
  acceptedFileTypes,
  invalidFileMode = 'show',
  onChange,
  onError,
  onClearError,
  workspace,
  errorText,
  allowMultiple = false,
  includeTabs = false,
  includeDataset = true,
  listLabel,
  inline = false,
  autoCommit = false,
  filesetPurpose,
  allowMultipleUpload = false,
  datasetLabel,
  autoSelectFirstAcceptable,
  showUpdatedAt,
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const files = useMemo(() => {
    if (!value) return [];
    return Array.isArray(value) ? value : [value];
  }, [value]);

  const {
    previewFile,
    previewContent,
    isLoadingPreview,
    previewError,
    setPreviewFile,
    clearPreview,
  } = useFilePreview();

  // Track blob URLs created for uploaded files to clean up later
  const blobUrlsRef = useRef<Set<string>>(new Set());

  // Clean up orphaned blob URLs when files change (e.g., file deleted or parent changes value)
  useEffect(() => {
    const currentFileUrls = new Set(files.map((f) => f.url).filter(Boolean));

    blobUrlsRef.current.forEach((url) => {
      if (!currentFileUrls.has(url)) {
        URL.revokeObjectURL(url);
        blobUrlsRef.current.delete(url);
      }
    });
  }, [files]);

  // Clean up all blob URLs on unmount
  useEffect(() => {
    const blobUrls = blobUrlsRef.current;
    return () => {
      blobUrls.forEach((url) => URL.revokeObjectURL(url));
      blobUrls.clear();
    };
  }, []);

  const datasetName = files[0]?.url
    ? (parseFilesetLocation(files[0].url)?.name ?? files[0].url)
    : '';

  const validateFiles = (filesToValidate: FileListItem[]): FileListItem[] =>
    filesToValidate.filter((file) => !isAcceptedFileType(file.path, acceptedFileTypes));

  const handleModalSubmit = async (data: SubmitUploadType) => {
    let newFiles: FileListItem[] = [];

    if (data.type === 'dataset') {
      const { dataset, path, url } = data;
      const fileItem: FileListItem = { dataset, path, url };
      newFiles = [fileItem];
    } else if (data.type === 'file' && data.files) {
      newFiles = await Promise.all(
        data.files.map(async (file) => {
          const content = await file.text();
          const blobUrl = URL.createObjectURL(file);
          blobUrlsRef.current.add(blobUrl);
          return {
            path: file.name,
            url: blobUrl,
            content,
          };
        })
      );
    }

    // Validate all files and collect invalid ones
    const invalidFiles = validateFiles(newFiles);
    if (invalidFiles.length > 0) {
      // Revoke any blob URLs we just created — keeping them around leaks
      // memory until unmount when the rejected selection is never adopted.
      newFiles.forEach((file) => {
        if (file.url?.startsWith('blob:')) {
          URL.revokeObjectURL(file.url);
          blobUrlsRef.current.delete(file.url);
        }
      });
      const invalidPaths = invalidFiles.map((f) => f.path).join(', ');
      const invalidExtensions = invalidFiles.map((f) => getFileExtension(f.path)).join(', ');
      const errorMessage = `Invalid file type(s) (${invalidExtensions}). Accepted types: ${acceptedFileTypes?.join(', ')}. Invalid files: ${invalidPaths}`;
      onError?.({ message: errorMessage, filepath: invalidPaths });
      setIsModalOpen(false);
      return;
    }

    onClearError?.();
    onChange(allowMultiple ? [...files, ...newFiles] : newFiles);
    setIsModalOpen(false);
  };

  const handleDeleteFile = (filepath: string) => {
    const newFiles = files.filter((f) => f.path !== filepath);
    onClearError?.();
    onChange(newFiles);
  };

  return (
    <>
      <Stack gap="density-sm">
        {inline ? (
          <InlineUploadPicker
            workspace={workspace}
            includeDataset={includeDataset}
            includeTabs={includeTabs}
            allowMultipleFileSelection={allowMultiple || allowMultipleUpload}
            acceptableFileTypes={acceptedFileTypes}
            invalidFileMode={invalidFileMode}
            onSubmit={handleModalSubmit}
            autoCommit={autoCommit}
            filesetPurpose={filesetPurpose}
            datasetLabel={datasetLabel}
            autoSelectFirstAcceptable={autoSelectFirstAcceptable}
            showUpdatedAt={showUpdatedAt}
          />
        ) : (
          <DatasetFileSelectButton
            datasetName={datasetName}
            onSelectClick={() => setIsModalOpen(true)}
            onChangeClick={() => setIsModalOpen(true)}
          />
        )}

        {files.length > 0 && !(inline && autoCommit) ? (
          <FileList
            label={listLabel}
            files={files}
            onDeleteFile={handleDeleteFile}
            onPreviewFile={setPreviewFile}
          />
        ) : null}

        {errorText ? (
          <Text kind="label/regular/sm" className="text-feedback-danger pt-density-xs">
            {errorText}
          </Text>
        ) : null}
      </Stack>

      {previewFile && (
        <SidePanel
          slotHeading={
            <div className="flex gap-2 items-center">
              <FolderOpen />
              {previewFile.dataset
                ? `${previewFile.dataset.workspace}/${previewFile.dataset.name}/${previewFile.path}`
                : previewFile.path}
            </div>
          }
          side="right"
          open
          onOpenChange={clearPreview}
          onEscapeKeyDown={(e) => {
            e.preventDefault();
            clearPreview();
          }}
          onPointerDownOutside={(e) => {
            e.preventDefault();
            clearPreview();
          }}
          attributes={{
            SidePanelHeading: { className: 'font-normal' },
          }}
          bordered
          modal
          className="max-w-[960px] w-full"
        >
          <FileContentPreview
            file={previewFile}
            content={previewContent ?? undefined}
            isLoading={isLoadingPreview}
            error={previewError}
          />
        </SidePanel>
      )}

      {!inline ? (
        <UploadModal
          includeDataset={includeDataset}
          includeTabs={includeTabs}
          workspace={workspace}
          open={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onSubmit={handleModalSubmit}
          acceptableFileTypes={acceptedFileTypes}
          invalidFileMode={invalidFileMode}
          allowMultipleFileSelection={allowMultiple || allowMultipleUpload}
          // Default modal width is too narrow for the files table; widen the
          // surface so file paths and sizes don't get truncated. Stays
          // responsive with ``max-w-[90vw]``.
          attributes={{ ModalContent: { className: 'w-[800px] max-w-[90vw]' } }}
        />
      ) : null}
    </>
  );
};
