// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Modal, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface DuplicateFileConfirmationModalProps {
  duplicateFiles: File[];
  onConfirm: () => void;
  onCancel: () => void;
  isPending: boolean;
}

export const DuplicateFileConfirmationModal: FC<DuplicateFileConfirmationModalProps> = ({
  duplicateFiles,
  onConfirm,
  onCancel,
  isPending,
}) => {
  return (
    <Modal
      open={duplicateFiles.length > 0}
      onOpenChange={(isOpen) => {
        if (!isOpen && !isPending) onCancel();
      }}
      slotHeading={
        <>Replace {duplicateFiles.length === 1 ? 'File' : `${duplicateFiles.length} Files`}</>
      }
      slotFooter={
        <Flex justify="end" gap="density-xs" align="center" className="w-full">
          <Button onClick={onCancel} kind="tertiary" color="neutral" disabled={isPending}>
            Cancel
          </Button>
          <Button color="brand" onClick={onConfirm} disabled={isPending}>
            {isPending ? 'Replacing...' : 'Replace'}
          </Button>
        </Flex>
      }
    >
      <Flex direction="col" gap="density-md">
        <Text kind="body/regular/md">
          {duplicateFiles.length === 1
            ? 'A file with this name already exists:'
            : `${duplicateFiles.length} files with these names already exist:`}
        </Text>
        <ul>
          {duplicateFiles.map((file) => (
            <li key={file.name}>
              <Text kind="body/bold/md">{file.name.split('/').pop()}</Text>
            </li>
          ))}
        </ul>
        <Text kind="body/regular/md">
          Do you want to replace {duplicateFiles.length === 1 ? 'it' : 'them'}?
        </Text>
      </Flex>
    </Modal>
  );
};
