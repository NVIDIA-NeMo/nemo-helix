// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ListItem } from '@nemo/common/src/components/FileList/ListItem';
import { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Divider, Text, Button, Flex, Stack } from '@nvidia/foundations-react-core';
import { IconDocument, Eye, Trash } from '@nvidia/react-gui-icons/icons';
import { FC } from 'react';

const numberFormatter = new Intl.NumberFormat('en-US');

export interface FileListItem {
  dataset?: FilesetOutput;
  path: string;
  url?: string;
  content?: string; // Optional: if provided, uses local content instead of fetching
  /**
   * Optional row/example count rendered to the right of the path. Used by
   * dataset pickers (e.g., the customizer fine-tune form) to surface how many
   * examples each file contributes to the final tally.
   */
  rowCount?: number;
}

export interface FileListError {
  error: string;
}

export interface FileListProps {
  files: (FileListItem | FileListError)[];
  onDeleteFile?: (filepath: string) => void;
  onPreviewFile?: (file: FileListItem) => void;
  allowPreview?: boolean;
  allowDelete?: boolean;
  label?: string;
}

export const FileList: FC<FileListProps> = ({
  files,
  onDeleteFile,
  onPreviewFile,
  allowPreview = true,
  allowDelete = true,
  label,
}) => {
  return (
    <Stack>
      {label && (
        <Text kind="label/bold/sm" className="mb-density-md">
          {label}
        </Text>
      )}
      {files.map((file, index) => {
        if ('error' in file) {
          return (
            <div key={index}>
              <Divider className="h-fit" />
              <ListItem error value={file.error} />
            </div>
          );
        }
        const filepath = file.path;
        return (
          <div key={filepath}>
            <Divider className="h-fit" />
            <ListItem
              value={filepath}
              startIconSlot={<IconDocument />}
              endIconSlot={
                <Flex align="center" gap="density-md">
                  {typeof file.rowCount === 'number' && (
                    <Text kind="body/regular/sm" className="text-fg-subdued whitespace-nowrap">
                      {numberFormatter.format(file.rowCount)} example
                      {file.rowCount === 1 ? '' : 's'}
                    </Text>
                  )}
                  <Flex align="center" gap="density-xs">
                    {allowPreview && (
                      <Button
                        onClick={() => onPreviewFile?.(file)}
                        aria-label="Preview file"
                        type="button"
                        kind="tertiary"
                        size="small"
                      >
                        <Eye width={16} height={16} />
                      </Button>
                    )}
                    {allowDelete && (
                      <Button
                        onClick={() => onDeleteFile?.(filepath)}
                        aria-label="Delete file"
                        type="button"
                        kind="tertiary"
                        size="small"
                      >
                        <Trash width={16} height={16} />
                      </Button>
                    )}
                  </Flex>
                </Flex>
              }
            />
            {index === files.length - 1 && <Divider className="h-fit" />}
          </div>
        );
      })}
    </Stack>
  );
};
