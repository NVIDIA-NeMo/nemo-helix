// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { FormField, Select, Stack } from '@nvidia/foundations-react-core';
import { isEvalConfigCandidate } from '@studio/components/evaluation/consts';
import { filesetNameOption } from '@studio/components/evaluation/shared/filesetNameOption';
import { type ReactElement } from 'react';
import {
  type Control,
  type FieldValues,
  type Path,
  useController,
  useWatch,
} from 'react-hook-form';

interface FilesetConfigPickerProps<T extends FieldValues> {
  workspace: string;
  control: Control<T>;
  /** Form field holding the chosen fileset name. */
  filesetName: Path<T>;
  /** Form field holding the chosen config's path within that fileset. */
  fileName: Path<T>;
  /** Fired whenever the fileset or the file changes, so the owner can drop state tied to the old pick. */
  onPick?: () => void;
  disabled?: boolean;
  /** The chosen file is still being read. */
  loading?: boolean;
  error?: string;
}

export function FilesetConfigPicker<T extends FieldValues>({
  workspace,
  control,
  filesetName,
  fileName,
  onPick,
  disabled,
  loading,
  error,
}: FilesetConfigPickerProps<T>): ReactElement {
  const fileset: string = useWatch({ control, name: filesetName });
  const { field: fileField } = useController({ control, name: fileName });

  const pickFile = (path: string) => {
    fileField.onChange(path);
    onPick?.();
  };

  const filesQuery = useFilesListFilesetFiles(workspace, fileset, undefined, {
    query: { enabled: !!fileset },
  });

  const fileItems = (filesQuery.data?.data ?? [])
    .map((file) => file.path)
    .filter(isEvalConfigCandidate)
    .sort()
    .map((path) => ({ value: path, children: path }));

  const noConfigFiles = !!fileset && filesQuery.isSuccess && fileItems.length === 0;
  const placeholder = filesQuery.isLoading || loading ? 'Loading files...' : 'Select a config';

  return (
    <Stack gap="density-sm">
      <FilesetSearchableSelect<T>
        workspace={workspace}
        useControllerProps={{ control, name: filesetName }}
        formFieldProps={{ slotLabel: 'Config Fileset' }}
        renderOption={filesetNameOption}
        onChange={() => pickFile('')}
        disabled={disabled}
      />
      <FormField
        slotLabel="Config File"
        slotHelp={
          noConfigFiles ? 'This fileset has no JSON or YAML files.' : 'A JSON or YAML config.'
        }
        slotError={error}
        status={error ? 'error' : undefined}
      >
        <Select
          disabled={disabled || !fileset}
          items={fileItems}
          value={fileField.value}
          onValueChange={pickFile}
          placeholder={placeholder}
        />
      </FormField>
    </Stack>
  );
}
