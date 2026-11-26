// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetCombobox } from '@nemo/common/src/components/FilesetPicker/FilesetCombobox';
import {
  DEFAULT_FILESET_PICKER_NOUN,
  type FilesetPickerNoun,
  type FilesetPickerPurpose,
} from '@nemo/common/src/components/FilesetPicker/types';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { FormField, Stack } from '@nvidia/foundations-react-core';
import { type FC, type ReactNode, useId, useState } from 'react';

export interface FilesetPickerProps {
  workspace: string;
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

export const FilesetPicker: FC<FilesetPickerProps> = ({
  workspace,
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
  const [missing, setMissing] = useState(false);
  const shownError =
    errorText ?? (missing ? `This ${noun.one} no longer exists. Select another one.` : undefined);
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
            purpose={purpose}
            value={value}
            onChange={onChange}
            noun={noun}
            onRequestNew={onRequestNew}
            disabled={disabled}
            onBlur={onBlur}
            onMissingChange={setMissing}
            status={status === 'error' ? 'error' : undefined}
          />
        )}
      </FormField>
    </Stack>
  );
};
