// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import {
  Banner,
  FormField,
  SegmentedControl,
  Select,
  Stack,
  Text,
  UploadInputElement,
  UploadRoot,
  UploadTrigger,
} from '@nvidia/foundations-react-core';
import type {
  BundleSourceKind,
  BundleSourceState,
  FilesetFormValues,
} from '@studio/components/BundleSourcePicker/types';
import { type ReactElement, type ReactNode, useCallback } from 'react';

const SOURCE_ITEMS: { value: BundleSourceKind; children: string }[] = [
  { value: 'upload', children: 'Upload from your computer' },
  { value: 'fileset', children: 'Choose from a fileset' },
];

const filesetOption = (fileset: { name: string }) => ({ value: fileset.name, label: fileset.name });

const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

export interface BundleSourcePickerProps<T> {
  workspace: string;
  /** From {@link useBundleSource}, which the owner keeps so it can read `active.selection`. */
  state: BundleSourceState<T>;
  /** Heading above the picker, e.g. "Optimize bundle". */
  title?: ReactNode;
  /** Info on the fileset field: what happens to the fileset once the bundle is used. */
  filesetInfo?: ReactNode;
  disabled?: boolean;
}

/**
 * Picks a bundle — one driving file plus the assets it references — uploaded from the user's
 * machine as a folder or already in a fileset, then the driving file in it, and shows read errors
 * and the problems `state.spec.validate` finds. What counts as the driving file comes entirely from the spec, so any
 * "bring your own config" flow can reuse this; see `useBundleSource`.
 */
export function BundleSourcePicker<T>({
  workspace,
  state,
  title,
  filesetInfo,
  disabled,
}: BundleSourcePickerProps<T>): ReactElement {
  const { spec, active, upload, fileset } = state;
  const fileLabel = capitalize(spec.label);

  const setFolderInput = useCallback((node: HTMLInputElement | null) => {
    // webkitdirectory is absent from React's input attribute types.
    node?.setAttribute('webkitdirectory', '');
  }, []);

  const showFileSelect = state.source === 'upload' ? !!upload.summary : !!fileset.fileset;
  const fileHelp =
    state.source === 'fileset' && fileset.hasNoCandidates
      ? `This fileset has no ${spec.candidateNoun}.`
      : undefined;

  return (
    <Stack gap="density-md">
      {title ? <Text kind="label/semibold/md">{title}</Text> : null}
      <SegmentedControl
        aria-label="Bundle source"
        size="tiny"
        className="w-full"
        value={state.source}
        onValueChange={(value) => state.setSource(value as BundleSourceKind)}
        items={SOURCE_ITEMS}
      />
      {state.source === 'upload' ? (
        <Stack gap="density-sm">
          <UploadRoot multiple disabled={disabled}>
            <UploadTrigger
              className="w-full"
              data-testid="bundle-dropzone"
              onDrop={(event) => {
                event.preventDefault();
                event.stopPropagation();
                if (disabled) return;
                const items = Array.from(event.dataTransfer.items);
                if (items.length > 0) upload.onItemsDropped(items);
              }}
              slotAnchor={upload.summary ? 'Choose a different folder' : 'Choose a folder'}
              slotHeaderText={` containing the ${spec.label}, or drop it here.`}
            >
              <UploadInputElement
                ref={setFolderInput}
                data-testid="bundle-input"
                multiple
                onChange={(event) => {
                  const files = Array.from(event.target.files ?? []);
                  event.target.value = '';
                  if (files.length > 0) upload.onFilesPicked(files);
                }}
              />
            </UploadTrigger>
          </UploadRoot>
          {upload.summary ? <Text kind="body/regular/sm">{upload.summary}</Text> : null}
        </Stack>
      ) : (
        <FilesetSearchableSelect<FilesetFormValues>
          workspace={workspace}
          useControllerProps={{ control: fileset.control, name: 'fileset' }}
          formFieldProps={{ slotLabel: 'Fileset', slotInfo: filesetInfo }}
          renderOption={filesetOption}
          onChange={() => fileset.setPath('')}
          disabled={disabled}
        />
      )}
      {showFileSelect ? (
        <FormField slotLabel={fileLabel} slotHelp={fileHelp}>
          <Select
            aria-label={fileLabel}
            value={active.path}
            onValueChange={active.setPath}
            disabled={disabled || active.candidatePaths.length === 0}
            placeholder={active.isLoading ? 'Loading...' : `Select the ${spec.label}`}
            items={active.candidatePaths.map((path) => ({ value: path, children: path }))}
          />
        </FormField>
      ) : null}
      {active.error ? (
        <Banner status="error" kind="inline" data-testid="bundle-error">
          <Text kind="body/regular/sm">{active.error}</Text>
        </Banner>
      ) : null}
      {active.problems.length > 0 ? (
        <Banner status="error" kind="inline" data-testid="bundle-problems">
          <Stack gap="density-xs">
            <Text kind="body/semibold/sm">
              {`${active.problems.length} ${active.problems.length === 1 ? 'problem' : 'problems'} in ${active.path}`}
            </Text>
            <ul className="list-disc pl-5">
              {active.problems.map((problem) => (
                <li key={problem}>
                  <Text kind="body/regular/sm">{problem}</Text>
                </li>
              ))}
            </ul>
          </Stack>
        </Banner>
      ) : null}
    </Stack>
  );
}
