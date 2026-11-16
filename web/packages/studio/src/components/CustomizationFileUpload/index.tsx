// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RenderFileTagFn } from '@nemo/common/src/components/FileUpload';
import {
  CustomizationFileTag,
  type CustomizationFileTagFile,
} from '@studio/components/CustomizationFileTag';
import { DatasetFileUpload } from '@studio/components/DatasetFileUpload';
import {
  CUSTOMIZATION_FILESET_FILE_PREFIXES,
  type CustomizationFileType,
} from '@studio/constants/customization';
import type { ComponentProps, FC } from 'react';

interface Props extends ComponentProps<typeof DatasetFileUpload> {
  customizationFileType: CustomizationFileType;
}
/**
 * A wrapper with specific UI for uploading files for a customization dataset.
 * Wraps the DatasetFileUpload component.
 * @returns
 */
export const CustomizationFileUpload: FC<Props> = ({
  customizationFileType,
  status,
  required,
  ...props
}) => {
  const renderFileTag: RenderFileTagFn = (file, disabled, onRemove) => {
    // TODO: in the future we may support uploading multiple training/validation files
    const customizationFiles: CustomizationFileTagFile[] = [
      {
        path: `${CUSTOMIZATION_FILESET_FILE_PREFIXES[customizationFileType]}/${file.name}`,
      },
    ];
    return (
      <CustomizationFileTag
        key={`file-tag-${file.name}`}
        files={customizationFiles}
        fileType={customizationFileType}
        total={customizationFiles.length}
        fileName={file.name}
        status={status}
        disabled={disabled}
        onClick={onRemove}
        required={required}
      />
    );
  };
  return (
    <DatasetFileUpload
      required={required}
      status={status}
      {...props}
      renderFileTag={renderFileTag}
    />
  );
};
