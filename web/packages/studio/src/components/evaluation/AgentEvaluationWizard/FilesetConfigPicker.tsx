// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { Stack } from '@nvidia/foundations-react-core';
import { isEvalConfigCandidate } from '@studio/components/evaluation/AgentEvaluationWizard/consts';
import { filesetNameOption } from '@studio/components/evaluation/utils';
import { type ReactElement } from 'react';
import {
  type Control,
  type FieldValues,
  type Path,
  type PathValue,
  useFormContext,
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
  const { setValue } = useFormContext<T>();

  const clearFile = () => {
    setValue(fileName, '' as PathValue<T, Path<T>>);
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
        onChange={clearFile}
        disabled={disabled}
      />
      <ControlledSelect
        useControllerProps={{ control, name: fileName }}
        formFieldProps={{
          slotLabel: 'Config File',
          slotHelp: noConfigFiles
            ? 'This fileset has no JSON or YAML files.'
            : 'A JSON or YAML config.',
          slotError: error,
          status: error ? 'error' : undefined,
        }}
        disabled={disabled || !fileset}
        items={fileItems}
        onChange={() => onPick?.()}
        placeholder={placeholder}
      />
    </Stack>
  );
}
