// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatFileSize } from '@nemo/common/src/components/UploadModal/utils';
import { Banner, Button, Stack, Text } from '@nvidia/foundations-react-core';
import type { CustomizationTransfer } from '@studio/util/customizationTransfer';
import type { FC } from 'react';

interface Props {
  transfer: CustomizationTransfer;
  onViewLogs?: () => void;
}

export const CustomizationTransferBanner: FC<Props> = ({ transfer, onViewLogs }) => {
  const { stepLabel, uploadedFiles, totalFiles, totalBytes } = transfer;

  const progress =
    uploadedFiles !== undefined && totalFiles !== undefined && totalFiles > 0
      ? `${uploadedFiles} of ${totalFiles} files uploaded` +
        (totalBytes ? ` (${formatFileSize(totalBytes)} in total)` : '')
      : undefined;

  return (
    <Banner
      kind="inline"
      status="info"
      className="shrink-0"
      data-testid="customization-transfer-banner"
      slotActions={
        onViewLogs ? (
          <Button kind="secondary" size="small" onClick={onViewLogs}>
            View {stepLabel} logs
          </Button>
        ) : undefined
      }
    >
      <Stack gap="density-xs">
        <Text kind="body/regular/sm">
          Training has completed. The model files are being transferred, so the job is still active.
        </Text>
        <Text kind="body/regular/sm" className="text-secondary">
          Large models can take a long time to upload.
          {progress && ` ${progress}; the count updates as each file finishes.`}
        </Text>
      </Stack>
    </Banner>
  );
};
