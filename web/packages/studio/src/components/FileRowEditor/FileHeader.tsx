// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Spinner, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import { Download, IconDocument, FolderOpen, Add, Floppy } from '@nvidia/react-gui-icons/icons';
import { FILE_FORMAT_TAG_COLOR } from '@studio/components/FileRowEditor/constants';
import type { DataFileFormat } from '@studio/components/FileRowEditor/parse';
import { type ChangeEvent, type FC, type ReactNode, type RefObject } from 'react';

export interface FileHeaderProps {
  fileName: string;
  slotFileName?: ReactNode;
  fileFormat: DataFileFormat;
  rowCount: number;
  columnCount: number;
  fileSizeLabel: string;
  /** Parse/load error to surface in place of the "Schema inferred" note. */
  loadError: string | null;
  /** Whether to show the "Open File" action that loads a local file in-browser. */
  showOpenFile: boolean;
  fileInputRef: RefObject<HTMLInputElement | null>;
  onFileSelected: (event: ChangeEvent<HTMLInputElement>) => void;
  onOpenFileClick: () => void;
  onDownload: () => void;
  onAddRow: () => void;
  /** Disables the Download action (e.g. when there are no rows to export). */
  downloadDisabled: boolean;
  /**
   * Persists the current rows to the backing store. When provided, a "Save File" action
   * is shown; when omitted, the editor stays in-memory only (e.g. the standalone demo).
   */
  onSaveFile?: () => void;
  /** Whether a save is in flight — shows a spinner and disables the Save action. */
  isSaving?: boolean;
  /** Disables the Save action (e.g. when there are no unsaved changes). */
  saveDisabled?: boolean;
  /** When set, disables the Save action and shows this text as its tooltip. */
  saveDisabledReason?: string;
  /** Whether there are staged edits not yet persisted — shows an "Unsaved changes" chip. */
  hasUnsavedChanges?: boolean;
}

/** Header summary + toolbar for the {@link FileRowEditor}: file identity, stats, actions. */
export const FileHeader: FC<FileHeaderProps> = ({
  fileName,
  slotFileName,
  fileFormat,
  rowCount,
  columnCount,
  fileSizeLabel,
  loadError,
  showOpenFile,
  fileInputRef,
  onFileSelected,
  onOpenFileClick,
  onDownload,
  onAddRow,
  downloadDisabled,
  onSaveFile,
  isSaving = false,
  saveDisabled = false,
  saveDisabledReason,
  hasUnsavedChanges = false,
}) => (
  <Flex align="center" gap="density-md" className="w-full shrink-0">
    <Flex align="center" justify="center" className="size-10 shrink-0 rounded-md bg-surface-sunken">
      <IconDocument className="size-20 text-secondary" />
    </Flex>
    <Stack gap="density-xs" className="min-w-0 flex-1">
      <Flex align="center" gap="density-sm" className="min-w-0">
        {slotFileName ?? (
          <Text kind="title/xs" className="truncate">
            {fileName}
          </Text>
        )}
        <Tag kind="solid" color={FILE_FORMAT_TAG_COLOR[fileFormat]} readOnly>
          {fileFormat === 'unknown' ? 'FILE' : fileFormat.toUpperCase()}
        </Tag>
      </Flex>
      <Flex align="center" gap="density-sm" className="text-secondary">
        <Text kind="body/regular/sm" className="text-secondary">
          {rowCount.toLocaleString()} rows · {columnCount} columns · {fileSizeLabel}
        </Text>
        <Text kind="body/regular/sm" className="text-secondary">
          ·
        </Text>
        {loadError ? (
          <Text kind="body/regular/sm" className="text-danger">
            {loadError}
          </Text>
        ) : (
          <Text kind="body/regular/sm" className="text-secondary">
            Schema inferred
          </Text>
        )}
      </Flex>
    </Stack>
    <Flex align="center" gap="density-sm" className="shrink-0">
      {showOpenFile && (
        <>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,.jsonl,.ndjson,.csv"
            className="hidden"
            aria-hidden="true"
            tabIndex={-1}
            onChange={onFileSelected}
          />
          <Button kind="secondary" color="neutral" onClick={onOpenFileClick}>
            <FolderOpen size="16" />
            Open File
          </Button>
        </>
      )}
      <Button kind="secondary" color="neutral" onClick={onDownload} disabled={downloadDisabled}>
        <Download size="16" />
        Download
      </Button>
      <Button
        kind={onSaveFile ? 'secondary' : 'primary'}
        color={onSaveFile ? 'neutral' : 'brand'}
        onClick={onAddRow}
      >
        <Add size="16" />
        Add Row
      </Button>
      {onSaveFile && (
        <span className="relative inline-flex">
          <Button
            kind="primary"
            color="brand"
            onClick={onSaveFile}
            disabled={saveDisabled || isSaving || Boolean(saveDisabledReason)}
            title={saveDisabledReason}
          >
            {isSaving ? <Spinner size="small" aria-label="Saving" /> : <Floppy size="16" />}
            Save File
          </Button>
          {hasUnsavedChanges && (
            <span
              title="Unsaved changes"
              className="pointer-events-none absolute -left-0.5 -top-0.5 size-2.5 rounded-full bg-feedback-danger ring-1 ring-surface-sunken dark:ring-surface-raised"
            >
              <span className="sr-only">Unsaved changes</span>
            </span>
          )}
        </span>
      )}
    </Flex>
  </Flex>
);
