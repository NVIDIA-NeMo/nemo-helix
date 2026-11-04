// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ActionsMenu } from '@nemo/common/src/components/ActionsMenu';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  type DownloadFormat,
  type DownloadPayload,
  downloadPayload,
} from '@nemo/common/src/utils/download';
import { Download } from '@nemo/icons/icons';
import { type FC, useState } from 'react';

interface DownloadMenuProps {
  /** File name without an extension. */
  fileName: string;
  /** Resolves the data at click time, so a page can fetch anything it does not already hold. */
  getData: () => DownloadPayload | Promise<DownloadPayload>;
  disabled?: boolean;
  label?: string;
  /** Overrides the default description of what each format contains. */
  descriptions?: Partial<Record<DownloadFormat, string>>;
}

const FORMATS: { format: DownloadFormat; label: string; description: string }[] = [
  { format: 'json', label: 'JSON', description: 'Full record with details and overall scores' },
  { format: 'jsonl', label: 'JSONL', description: 'One result per line, for scripts and datasets' },
  { format: 'csv', label: 'CSV', description: 'Results as a table, for spreadsheets' },
];

/** A "Download" menu offering the given data as JSON, JSONL, or CSV. */
export const DownloadMenu: FC<DownloadMenuProps> = ({
  fileName,
  getData,
  disabled,
  label = 'Download',
  descriptions,
}) => {
  const toast = useToast();
  const [isDownloading, setIsDownloading] = useState(false);

  const download = async (format: DownloadFormat) => {
    setIsDownloading(true);
    try {
      if (!downloadPayload(await getData(), fileName, format)) {
        toast.warning('There are no results to download.');
      }
    } catch (error) {
      toast.error(
        `Failed to download: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <ActionsMenu
      label={isDownloading ? 'Downloading…' : label}
      icon={<Download />}
      kind="secondary"
      disabled={disabled || isDownloading}
      ariaLabel="Download results"
      data-testid="download-menu"
      actions={FORMATS.map(({ format, label, description }) => ({
        label,
        description: descriptions?.[format] ?? description,
        onSelect: () => void download(format),
      }))}
    />
  );
};
