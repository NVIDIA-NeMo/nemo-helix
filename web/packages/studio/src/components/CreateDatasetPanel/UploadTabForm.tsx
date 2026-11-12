// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Block, Button, Flex, FormField, Stack, Text } from '@nvidia/foundations-react-core';
import type { FilesetEntry } from '@studio/api/files/types';
import { DescriptionField } from '@studio/components/CreateDatasetPanel/DescriptionField';
import type { Draft, PanelAction } from '@studio/components/CreateDatasetPanel/draft';
import { FileRow } from '@studio/components/CreateDatasetPanel/FileRow';
import { NameField } from '@studio/components/CreateDatasetPanel/NameField';
import type { DraftErrors } from '@studio/components/CreateDatasetPanel/validateDraft';
import { type ChangeEvent, type Dispatch, type DragEvent, useCallback, useRef } from 'react';

const FOLDER_PREVIEW_COUNT = 5;

interface UploadTabFormProps {
  draft: Draft;
  /** Only the errors that should currently be on screen. */
  errors: DraftErrors;
  dispatch: Dispatch<PanelAction>;
}

/** A folder-selected file keeps its relative path, including the picked top-level folder (spec 10). */
const toEntries = (files: Iterable<File>): FilesetEntry[] =>
  Array.from(files, (file) => ({ path: file.webkitRelativePath || file.name, file }));

/** Dropped files only: a dropped folder arrives as a bogus file, and folders are picked with Choose folder. */
const droppedFiles = (data: DataTransfer): File[] =>
  Array.from(data.items)
    .filter((item) => item.kind === 'file' && item.webkitGetAsEntry()?.isDirectory !== true)
    .flatMap((item) => item.getAsFile() ?? []);

/** The Upload tab: stage files or a folder, name the dataset, describe it. Nothing uploads until Create. */
export function UploadTabForm({ draft, errors, dispatch }: UploadTabFormProps) {
  const filesInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement | null>(null);
  // A stable ref callback, so React does not detach and re-run it on every render.
  const setFolderInput = useCallback((node: HTMLInputElement | null) => {
    folderInput.current = node;
    // webkitdirectory is absent from React's input attribute types.
    node?.setAttribute('webkitdirectory', '');
  }, []);

  const { looseFiles, folderFiles } = draft;
  const folderName = folderFiles[0]?.path.split('/')[0];
  const hiddenFolderCount = Math.max(folderFiles.length - FOLDER_PREVIEW_COUNT, 0);

  const pick = (event: ChangeEvent<HTMLInputElement>, type: 'files-picked' | 'folder-picked') => {
    const entries = toEntries(event.target.files ?? []);
    // Clear so picking the same selection again still fires a change.
    event.target.value = '';
    // Dismissing the dialog can report no files; that must not wipe the existing selection.
    if (entries.length > 0) dispatch({ type, entries });
  };

  const drop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    const entries = toEntries(droppedFiles(event.dataTransfer));
    if (entries.length > 0) dispatch({ type: 'files-picked', entries });
  };

  return (
    <Stack gap="density-lg" className="w-full min-w-0">
      <FormField
        slotLabel="Files"
        slotHelp="Uploaded to platform default storage when you create the dataset."
        slotError={errors.files}
        status={errors.files ? 'error' : undefined}
        labelable={false}
      >
        <Stack gap="density-md" className="w-full">
          {/* Dropping is a pointer-only shortcut; Choose files is the keyboard path. */}
          <Block
            className="w-full rounded-md border border-dashed p-4"
            onDragOver={(event) => event.preventDefault()}
            onDrop={drop}
          >
            <Flex direction="col" align="center" gap="density-md">
              <Text kind="body/regular/md">Drag and drop files here</Text>
              {!folderName && (
                <Flex gap="density-md">
                  <Button
                    type="button"
                    kind="secondary"
                    onClick={() => filesInput.current?.click()}
                  >
                    Choose files
                  </Button>
                  <Button
                    type="button"
                    kind="secondary"
                    onClick={() => folderInput.current?.click()}
                  >
                    Choose folder
                  </Button>
                </Flex>
              )}
            </Flex>
          </Block>

          {looseFiles.map(({ path }) => (
            <FileRow
              key={path}
              path={path}
              onRemove={() => dispatch({ type: 'file-removed', path })}
            />
          ))}

          {folderName && (
            <Stack gap="density-sm" className="w-full rounded-md border p-3">
              <Flex align="center" justify="between" gap="density-md">
                <Text kind="label/bold/md">
                  {folderName} ({folderFiles.length} {folderFiles.length === 1 ? 'file' : 'files'})
                </Text>
                <Button
                  type="button"
                  kind="tertiary"
                  size="small"
                  aria-label={`Remove folder ${folderName}`}
                  onClick={() => dispatch({ type: 'folder-removed' })}
                >
                  Remove
                </Button>
              </Flex>
              {folderFiles.slice(0, FOLDER_PREVIEW_COUNT).map(({ path }) => (
                <Text key={path} kind="body/regular/sm" className="break-all">
                  {path}
                </Text>
              ))}
              {hiddenFolderCount > 0 && (
                <Text kind="body/regular/sm">and {hiddenFolderCount} more</Text>
              )}
            </Stack>
          )}
        </Stack>
      </FormField>

      <input
        ref={filesInput}
        type="file"
        aria-label="Files to upload"
        multiple
        hidden
        data-testid="create-dataset-files-input"
        onChange={(event) => pick(event, 'files-picked')}
      />
      <input
        ref={setFolderInput}
        type="file"
        aria-label="Folder to upload"
        hidden
        data-testid="create-dataset-folder-input"
        onChange={(event) => pick(event, 'folder-picked')}
      />

      <NameField value={draft.name} error={errors.name} dispatch={dispatch} />
      <DescriptionField value={draft.description} dispatch={dispatch} />
    </Stack>
  );
}
