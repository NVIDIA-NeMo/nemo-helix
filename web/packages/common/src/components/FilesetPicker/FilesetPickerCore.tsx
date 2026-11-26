// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ListFilesets, RetrieveFileset } from '@nemo/common/src/api/filesets/filesetPages';
import { FilesetCombobox } from '@nemo/common/src/components/FilesetPicker/FilesetCombobox';
import {
  DEFAULT_FILESET_PICKER_NOUN,
  type FilesetPickerNoun,
  type FilesetPickerPurpose,
  type FilesetSelectionError,
} from '@nemo/common/src/components/FilesetPicker/types';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { FormField, Stack } from '@nvidia/foundations-react-core';
import { type FC, type ReactNode, useId, useState } from 'react';

export interface FilesetPickerCoreProps {
  workspace: string;
  listFilesets: ListFilesets;
  retrieveFileset: RetrieveFileset;
  /** Selected fileset reference (`workspace/name`), or null. */
  value: string | null;
  onChange: (value: string | null, fileset?: FilesetOutput) => void;
  label: ReactNode;
  purpose?: FilesetPickerPurpose;
  noun?: FilesetPickerNoun;
  /** Adds a pinned "New <noun>" row; the consumer renders its own create flow. */
  onRequestNew?: () => void;
  disabled?: boolean;
  required?: boolean;
  slotHelp?: ReactNode;
  slotInfo?: ReactNode;
  errorText?: string;
  onBlur?: () => void;
}

const SELECTION_ERROR_MESSAGES: Record<FilesetSelectionError, (noun: string) => string> = {
  missing: (noun) => `This ${noun} no longer exists. Select another one.`,
  unavailable: (noun) => `Couldn't load the selected ${noun}. Try again later.`,
};

/**
 * Takes its fetchers so plugins can run it on `host.sdk`; Studio code uses `FilesetPicker`.
 */
export const FilesetPickerCore: FC<FilesetPickerCoreProps> = ({
  workspace,
  listFilesets,
  retrieveFileset,
  value,
  onChange,
  label,
  purpose,
  noun = DEFAULT_FILESET_PICKER_NOUN,
  onRequestNew,
  disabled,
  required,
  slotHelp,
  slotInfo,
  errorText,
  onBlur,
}) => {
  const id = useId();
  const [selectionError, setSelectionError] = useState<FilesetSelectionError | null>(null);
  const shownError =
    errorText ?? (selectionError ? SELECTION_ERROR_MESSAGES[selectionError](noun.one) : undefined);
  return (
    <Stack gap="3">
      <FormField
        id={id}
        slotLabel={label}
        slotHelp={slotHelp}
        slotInfo={slotInfo}
        slotError={shownError}
        status={shownError ? 'error' : undefined}
        required={required}
      >
        {({ status }) => (
          <FilesetCombobox
            id={id}
            workspace={workspace}
            listFilesets={listFilesets}
            retrieveFileset={retrieveFileset}
            purpose={purpose}
            value={value}
            onChange={onChange}
            noun={noun}
            onRequestNew={onRequestNew}
            disabled={disabled}
            onBlur={onBlur}
            onSelectionErrorChange={setSelectionError}
            status={status === 'error' ? 'error' : undefined}
          />
        )}
      </FormField>
    </Stack>
  );
};
