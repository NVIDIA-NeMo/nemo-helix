// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { triggerDownload } from '@nemo/common/src/utils/file';
import { logger } from '@nemo/common/src/utils/logger';
import { useDownloadFileAsArrayBuffer } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import type { FileSystemFile } from '@studio/components/FilesTable/utils';
import { getFileNameFromPath } from '@studio/util/files';
import { useCallback, useState } from 'react';

export interface UseBulkDownloadOptions {
  workspace: string;
  datasetName: string;
}

export interface UseBulkDownloadResult {
  handleBulkDownload: (files: FileSystemFile[]) => Promise<void>;
  isDownloading: boolean;
}

/**
 * Downloads multiple files by fetching each as an ArrayBuffer in parallel and
 * triggering a browser save per file (matching single-file behavior). A single
 * working toast tracks the batch; per-file failures are aggregated into one
 * partial/total-failure toast.
 */
export function useBulkDownload(options: UseBulkDownloadOptions): UseBulkDownloadResult {
  const { workspace, datasetName } = options;
  const toast = useToast();
  const downloadAsArrayBuffer = useDownloadFileAsArrayBuffer();
  const [isDownloading, setIsDownloading] = useState(false);

  const downloadOne = useCallback(
    async (path: string): Promise<boolean> => {
      const arrayBuffer = await downloadAsArrayBuffer({ workspace, datasetName, path });
      if (!arrayBuffer) return false;
      triggerDownload(arrayBuffer, getFileNameFromPath(path));
      return true;
    },
    [downloadAsArrayBuffer, workspace, datasetName]
  );

  const handleBulkDownload = useCallback(
    async (files: FileSystemFile[]) => {
      if (files.length === 0) return;

      setIsDownloading(true);
      const toastId = toast.workingWithId(
        files.length === 1 ? 'Downloading file...' : `Downloading ${files.length} files...`
      );

      try {
        const results = await Promise.all(files.map((file) => downloadOne(file.path)));
        toast.dismissToast(toastId);

        const successCount = results.filter(Boolean).length;
        const failCount = results.length - successCount;

        if (failCount === 0) {
          toast.success(
            successCount === 1
              ? 'Successfully downloaded file!'
              : `Successfully downloaded ${successCount} files!`
          );
        } else if (successCount === 0) {
          toast.error('Unable to download files. Please try again later.');
        } else {
          toast.error(
            `Downloaded ${successCount} of ${results.length} files. ${failCount} failed.`
          );
        }
      } catch (err) {
        logger.error('Bulk download failed', err);
        toast.dismissToast(toastId);
        toast.error(files.length === 1 ? 'Failed to download file' : 'Failed to download files');
      } finally {
        setIsDownloading(false);
      }
    },
    [downloadOne, toast]
  );

  return { handleBulkDownload, isDownloading };
}
