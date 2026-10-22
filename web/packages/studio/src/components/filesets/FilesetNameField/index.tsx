// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { useWatchedFilesetNameStatus } from '@studio/components/filesets/hooks/useFilesetNameStatus';
import { nameFieldSlots, type NameCheckStatus } from '@studio/components/form/entityNameCheck';
import type { ReactElement } from 'react';
import type { Control, FieldValues, Path } from 'react-hook-form';

interface FilesetNameFieldProps<T extends FieldValues> {
  control: Control<T>;
  name: Path<T>;
  workspace: string;
  /** A fileset this form already created, which the check must not count as a conflict. */
  ownName?: string;
  onStatusChange: (status: NameCheckStatus) => void;
  label?: string;
  disabled?: boolean;
  autoFocus?: boolean;
}

/** The name input of a Create Fileset form. Checks the name is free as the user types and
 *  reports a conflict ahead of the field's own format errors. */
export const FilesetNameField = <T extends FieldValues>({
  control,
  name,
  workspace,
  ownName,
  onStatusChange,
  label = 'Name',
  disabled,
  autoFocus,
}: FilesetNameFieldProps<T>): ReactElement => {
  const { candidate, status } = useWatchedFilesetNameStatus({
    control,
    name,
    workspace,
    ownName,
    onStatusChange,
  });

  return (
    <ControlledTextInput
      label={label}
      disabled={disabled}
      autoFocus={autoFocus}
      useControllerProps={{ name, control }}
      formFieldProps={nameFieldSlots({
        entity: 'fileset',
        preview: candidate,
        status,
        describe: (
          <>
            Fileset names must be unique within the{' '}
            <span className="text-primary">{workspace}</span> workspace.
          </>
        ),
        workspace,
      })}
    />
  );
};
