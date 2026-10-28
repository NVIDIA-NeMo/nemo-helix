// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CheckCircle, Close, Refresh, IconError } from '@nemo/icons/icons';
import { Button, Flex, Tag, TagProps } from '@nvidia/foundations-react-core';
import { FC, ReactNode, MouseEventHandler } from 'react';

export type FileTagStatus = 'success' | 'pending' | 'error' | 'idle';

export interface FileTagProps extends Omit<TagProps, 'style'> {
  fileName?: string;
  status?: FileTagStatus;
  noFileText?: string;
  required?: boolean;
  onNoFileClick?: () => void;
  slotStart?: ReactNode;
  className?: string;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  disabled?: boolean;
}

/**
 * Tag component for displaying file names with optional status indicators.
 * Used in forms and file upload contexts. Supports required state and custom click handlers.
 */
export const FileTag: FC<FileTagProps> = ({
  fileName,
  status,
  noFileText = 'No file detected',
  className,
  required,
  onNoFileClick,
  slotStart,
  disabled,
  onClick,
  ...tagProps
}) => {
  const missingFileNameChip = required ? (
    <p className="text-feedback-danger">{noFileText}</p>
  ) : (
    <Flex gap="density-sm">{noFileText}</Flex>
  );
  return (
    <Flex gap="density-xs" align="center" className={className}>
      {status === 'success' && (
        <CheckCircle color="var(--text-color-feedback-success)" width="16" height="16" />
      )}
      {status === 'pending' && <Refresh width="16" height="16" />}
      {status === 'error' && (
        <IconError color="var(--text-color-feedback-danger)" width="16" height="16" />
      )}
      {fileName ? (
        <Tag
          className="cursor-inherit"
          color="gray"
          kind="outline"
          type="button"
          {...tagProps}
          disabled={disabled}
          onClick={onClick}
        >
          {slotStart}
          <Flex gap="density-sm">{fileName}</Flex>
          {onClick && <Close />}
        </Tag>
      ) : onNoFileClick ? (
        <Button kind="tertiary" onClick={onNoFileClick} className="h-auto p-0">
          {missingFileNameChip}
        </Button>
      ) : (
        <div>{missingFileNameChip}</div>
      )}
    </Flex>
  );
};
