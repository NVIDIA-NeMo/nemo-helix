// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import {
  useFilesListFilesetFiles,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import {
  Flex,
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
} from '@nvidia/foundations-react-core';
import {
  FeedbackAddToDatasetFileSource,
  LOADING_FILES_OPTION,
} from '@studio/components/DatasetFileSelect/constants';
import { MultiselectOption } from '@studio/constants/mutliselect';
import { Plus } from 'lucide-react';
import { FC, ReactNode, useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

interface Props {
  datasetId: string;
  disabled?: boolean;
  helperText?: string;
  errorText?: ReactNode;
  hideLabel?: boolean;
  hideNew?: boolean;
  onResetFile?: () => void;
}

const fileToOption = (file: FilesetFileOutput): { value: string; label: string } => ({
  label: file.path,
  value: file.path,
});

export const DatasetFileSelect: FC<Props> = ({
  datasetId,
  disabled,
  helperText,
  errorText,
  hideLabel,
  hideNew,
  onResetFile,
}) => {
  const { setValue, control } = useFormContext();
  const selection = useWatch({ control, name: 'filepath' });
  const parts = getPartsFromReference(datasetId);
  const {
    data: dataset,
    isLoading: isLoadingDataset,
    isError: isDatasetError,
  } = useFilesRetrieveFileset(parts.workspace ?? '', parts.name ?? '', {
    query: { enabled: !!(parts.workspace && parts.name) },
  });
  const {
    data: filesResponse,
    isLoading: isLoadingFiles,
    isError: isFilesError,
  } = useFilesListFilesetFiles(dataset?.workspace ?? '', dataset?.name ?? '', undefined, {
    query: { enabled: !!(dataset?.workspace && dataset?.name) },
  });
  const files = filesResponse?.data;

  const isError = isDatasetError || isFilesError;
  const isDisabled = disabled || !datasetId || isLoadingDataset || isLoadingFiles;

  const fileOptions: MultiselectOption[] = useMemo(() => {
    if (isLoadingDataset || isLoadingFiles) {
      return [LOADING_FILES_OPTION];
    }

    return files?.map(fileToOption) || [];
  }, [files, isLoadingDataset, isLoadingFiles]);

  const handleFileSelect = (fileOption: string) => {
    if (fileOption === 'Other') {
      setValue('fileSource', FeedbackAddToDatasetFileSource.New, { shouldDirty: true });
      if (onResetFile) {
        onResetFile();
      }
    } else {
      setValue('fileSource', FeedbackAddToDatasetFileSource.Existing, {
        shouldDirty: true,
      });
      setValue('filepath', fileOption, { shouldDirty: true, shouldValidate: true });
    }
  };

  const getPlaceholder = () => {
    if (isLoadingFiles || isLoadingDataset) {
      return 'Loading files...';
    }

    return isDisabled ? 'Select a dataset above to view its files' : 'Select a file';
  };

  return (
    <Flex direction="col" gap="density-xs">
      <FormField
        slotLabel={!hideLabel && 'File name'}
        required
        slotHelp={helperText}
        slotError={errorText}
        status={control.getFieldState('filepath').invalid ? 'error' : undefined}
      >
        {({ status }) => (
          <SelectRoot
            onValueChange={handleFileSelect}
            value={selection ? selection : ''}
            disabled={isDisabled}
          >
            <SelectTrigger
              placeholder={getPlaceholder()}
              status={status}
              aria-label="dataset-file-select"
            />
            <SelectContent>
              <SelectListbox>
                {isError && (
                  <SelectItem value="" disabled>
                    Error loading files
                  </SelectItem>
                )}
                {!isDisabled && !isError && (
                  <>
                    {fileOptions.map((fileOpt, idx) => (
                      <SelectItem key={idx} value={fileOpt.value} className="kui-select-item">
                        {fileOpt.value}
                      </SelectItem>
                    ))}
                    {!hideNew && (
                      <SelectItem
                        className="border-t border-interaction-primary-base"
                        value="Other"
                      >
                        <Flex gap="density-md">
                          <Plus /> Create New File
                        </Flex>
                      </SelectItem>
                    )}
                  </>
                )}
              </SelectListbox>
            </SelectContent>
          </SelectRoot>
        )}
      </FormField>
    </Flex>
  );
};
