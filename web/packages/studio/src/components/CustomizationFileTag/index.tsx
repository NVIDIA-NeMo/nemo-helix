// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileTag, type FileTagProps } from '@nemo/common/src/components/FileTag';
import { Stack } from '@nvidia/foundations-react-core';
import {
  CUSTOMIZATION_FILESET_FILE_ICONS,
  CUSTOMIZATION_FILESET_FILE_LABELS,
  type CustomizationFileType,
} from '@studio/constants/customization';
import type { FC, MouseEventHandler } from 'react';

export interface CustomizationFileTagFile {
  path: string;
}
export interface CustomizationFileTagProps extends Omit<FileTagProps, 'noFileText' | 'slotStart'> {
  fileType: CustomizationFileType;
  files: CustomizationFileTagFile[];
  onNoFileClick?: () => void;
  total: number;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  disabled?: boolean;
}

export const CustomizationFileTag: FC<CustomizationFileTagProps> = ({
  fileType,
  files,
  total,
  onNoFileClick,
  onClick,
  disabled,
  ...fileTagProps
}) => {
  if (!files || files.length === 0) {
    return (
      <FileTag
        fileName=""
        noFileText={`No ${CUSTOMIZATION_FILESET_FILE_LABELS[fileType].toLowerCase()} detected`}
        slotStart={CUSTOMIZATION_FILESET_FILE_ICONS[fileType]}
        onNoFileClick={onNoFileClick}
        onClick={onClick}
        disabled={disabled}
        {...fileTagProps}
      />
    );
  }

  return (
    <Stack direction="row" gap="density-md" align="end">
      {files.map((file) => (
        <FileTag
          {...fileTagProps}
          key={file.path}
          fileName={file.path}
          slotStart={CUSTOMIZATION_FILESET_FILE_ICONS[fileType]}
          onClick={onClick}
          disabled={disabled}
        />
      ))}
      {total > files.length && <div>+{total - files.length} more</div>}
    </Stack>
  );
};
