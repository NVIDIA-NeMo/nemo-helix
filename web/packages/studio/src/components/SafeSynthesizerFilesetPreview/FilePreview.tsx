// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { triggerDownload } from '@nemo/common/src/utils/file';
import { Download, MoreVert } from '@nemo/icons/icons';
import {
  Button,
  DropdownContent,
  DropdownItem,
  DropdownRoot,
  DropdownTrigger,
  Flex,
  SidePanel,
  Spinner,
  Text,
} from '@nvidia/foundations-react-core';
import { FC, ReactNode } from 'react';

interface CSVPreviewProps {
  onClose: () => void;
  title: string;
  isLoading: boolean;
  error?: string;
  children: ReactNode;
  onDownload?: () => Promise<Blob | null>;
  downloadFileName?: string;
}

export const FilePreview: FC<CSVPreviewProps> = ({
  onClose,
  title,
  isLoading,
  error,
  children,
  onDownload,
  downloadFileName,
}) => {
  const toast = useToast();

  const handleDownload = async () => {
    if (!onDownload) return;

    try {
      const blob = await onDownload();
      if (blob) {
        const arrayBuffer = await blob.arrayBuffer();
        triggerDownload(arrayBuffer, downloadFileName || 'download.csv');
        toast.success('Successfully downloaded file!');
      } else {
        toast.error('Unable to download file. Please try again later.');
      }
    } catch {
      toast.error('Unable to download file. Please try again later.');
    }
  };

  return (
    <SidePanel
      slotHeading={title}
      side="right"
      open
      onOpenChange={onClose}
      onEscapeKeyDown={(e) => {
        e.preventDefault();
        onClose();
      }}
      onPointerDownOutside={(e) => {
        e.preventDefault();
        onClose();
      }}
      attributes={{
        SidePanelHeading: { className: 'font-normal' },
      }}
      bordered
      modal
      className="max-w-[960px] w-full"
    >
      {onDownload && (
        <DropdownRoot>
          <DropdownTrigger asChild showChevron={false}>
            <Button
              kind="tertiary"
              aria-label="Open file actions menu"
              className="absolute top-4 right-19"
            >
              <MoreVert />
            </Button>
          </DropdownTrigger>
          <DropdownContent align="end">
            <DropdownItem onClick={handleDownload}>
              <Flex align="center" gap="density-sm">
                <Download size="20" fill="solid" />
                Download
              </Flex>
            </DropdownItem>
          </DropdownContent>
        </DropdownRoot>
      )}
      {isLoading ? (
        <Flex align="center" justify="center" className="h-full">
          <Spinner size="medium" aria-label="Loading..." />
        </Flex>
      ) : error ? (
        <Flex align="center" justify="center" className="h-full">
          <Text>{error}</Text>
        </Flex>
      ) : (
        children
      )}
    </SidePanel>
  );
};
