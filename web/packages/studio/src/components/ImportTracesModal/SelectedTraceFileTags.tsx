// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileTag } from '@nemo/common/src/components/FileTag';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { FORMAT_LABELS } from '@studio/components/ImportTracesModal/detectTraceFormat';
import type { SelectedTraceFile } from '@studio/components/ImportTracesModal/ingestTraceFiles';
import { type FC } from 'react';

interface SelectedTraceFileTagsProps {
  files: SelectedTraceFile[];
  onRemove: (id: string) => void;
  disabled?: boolean;
}

/**
 * The picked files as removable chips, each labelled with the ingest format detected for it.
 * Showing the format here is what makes a mixed selection legible before anything is sent.
 */
export const SelectedTraceFileTags: FC<SelectedTraceFileTagsProps> = ({
  files,
  onRemove,
  disabled,
}) => {
  const undetected = files.filter(({ detection }) => detection.format === null);

  return (
    <Stack gap="density-md">
      <Flex gap="density-md" className="flex-wrap">
        {files.map(({ id, label, detection }) => (
          <FileTag
            key={id}
            fileName={
              detection.format ? `${label} · ${FORMAT_LABELS[detection.format]}` : `${label} · ?`
            }
            status={detection.format ? 'success' : 'error'}
            aria-label={`Remove ${label}`}
            disabled={disabled}
            onClick={() => onRemove(id)}
          />
        ))}
      </Flex>

      {undetected.map(({ id, label, detection }) => (
        <Text key={id} kind="body/regular/xs" className="text-feedback-danger">
          {label}: {detection.format === null ? detection.message : ''}
        </Text>
      ))}
    </Stack>
  );
};
