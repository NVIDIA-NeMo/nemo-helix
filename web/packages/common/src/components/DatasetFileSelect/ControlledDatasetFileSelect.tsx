// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type AcceptedFileType,
  DatasetFileSelect,
} from '@nemo/common/src/components/DatasetFileSelect/DatasetFileSelect';
import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { parseFilesetUrl } from '@nemo/common/src/components/DatasetFileSelect/utils';
import type { FileListItem } from '@nemo/common/src/components/FileList';
import type { UseControllerComponentProps } from '@nemo/common/src/utils/types';
import type { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { FormField } from '@nvidia/foundations-react-core';
import { FC, useMemo } from 'react';
import { useController } from 'react-hook-form';

interface ControlledDatasetFileSelectProps extends UseControllerComponentProps {
  label?: string;
  acceptedFileTypes?: AcceptedFileType[];
  /** How to render existing files whose extension isn't in
   *  ``acceptedFileTypes``. ``'show'`` (default) renders everything;
   *  ``'hide'`` filters them out; ``'disable'`` renders them but blocks
   *  selection. */
  invalidFileMode?: 'show' | 'hide' | 'disable';
  setError?: (error: { type?: string; message?: string }) => void;
  clearError?: () => void;
  workspace: string;
  /** The label for the file list. */
  listLabel?: string;
  /** When true, the picker UI renders inline (no "Select File" button + no
   *  secondary modal layer). */
  inline?: boolean;
  /** Inline-only: skip the "Add" button and commit on selection; also hides
   *  the file list rendered below the picker. */
  autoCommit?: boolean;
  /** Fileset ``purpose`` the picker lists. Defaults to ``'dataset'``. */
  filesetPurpose?: FilesetPurpose;
  /** Label for the fileset picker. Defaults to ``'Dataset'``. */
  datasetLabel?: string;
  /** Auto-select the first root-level accepted file on fileset selection. */
  autoSelectFirstAcceptable?: boolean;
  showUpdatedAt?: boolean;
  /**
   * Callback fired when a file is selected. Useful for custom validation or processing.
   * Called with the selected file info, or null when file is cleared.
   */
  onFileSelected?: (file: FileListItem | null) => void;
}

/**
 * A form-controlled wrapper around DatasetFileSelect that integrates with React Hook Form.
 *
 * @example
 * ```tsx
 * const form = useForm({
 *   defaultValues: {
 *     datasetFile: null // or 'hf://datasets/my-org/my-dataset/train.csv'
 *   }
 * });
 *
 * <ControlledDatasetFileSelect
 *   label="Training Data"
 *   useControllerProps={{ name: 'datasetFile', control: form.control }}
 *   acceptedFileTypes={['.jsonl', '.csv']}
 *   setError={(error) => form.setError('datasetFile', error)}
 *   clearError={() => form.clearErrors('datasetFile')}
 *   workspace={workspace}
 * />
 * ```
 */
export const ControlledDatasetFileSelect: FC<ControlledDatasetFileSelectProps> = ({
  label,
  acceptedFileTypes,
  invalidFileMode = 'show',
  setError,
  clearError,
  useControllerProps,
  formFieldProps,
  workspace,
  onFileSelected,
  listLabel,
  inline,
  autoCommit,
  filesetPurpose,
  datasetLabel,
  autoSelectFirstAcceptable,
  showUpdatedAt,
}) => {
  const {
    field: { onChange, value },
    fieldState: { error },
  } = useController(useControllerProps);

  // Convert form value (workspace/name#path) to SelectedFile (component value)
  const selectedFile = useMemo(() => {
    if (!value || typeof value !== 'string') {
      return null;
    }
    const parsed = parseFilesetLocation(value);
    if (!parsed?.objectPath) return null;
    return { path: parsed.objectPath, url: value };
  }, [value]);

  // Convert SelectedFile to workspace/name#path format when component changes
  const handleChange = (files: FileListItem[]) => {
    if (files.length === 0) {
      onChange(null);
      onFileSelected?.(null);
      return;
    }
    const file = files[0];
    const parsed = parseFilesetUrl(file.url ?? '');
    onChange(parsed ? `${parsed.workspace}/${parsed.name}#${parsed.path}` : null);
    onFileSelected?.(file);
  };

  const handleError = (error: { message: string; filepath?: string }) => {
    setError?.({ type: 'custom', message: error.message });
  };

  const handleClearError = () => {
    clearError?.();
  };

  return (
    <FormField
      name={useControllerProps.name}
      slotLabel={label}
      aria-label={label}
      status={error ? 'error' : undefined}
      slotError={error?.message?.toString()}
      {...formFieldProps}
    >
      <DatasetFileSelect
        value={selectedFile as FileListItem | FileListItem[] | null}
        acceptedFileTypes={acceptedFileTypes}
        invalidFileMode={invalidFileMode}
        onChange={handleChange}
        onError={handleError}
        onClearError={handleClearError}
        error={error?.message?.toString()}
        workspace={workspace}
        listLabel={listLabel}
        inline={inline}
        autoCommit={autoCommit}
        filesetPurpose={filesetPurpose}
        datasetLabel={datasetLabel}
        autoSelectFirstAcceptable={autoSelectFirstAcceptable}
        showUpdatedAt={showUpdatedAt}
      />
    </FormField>
  );
};
