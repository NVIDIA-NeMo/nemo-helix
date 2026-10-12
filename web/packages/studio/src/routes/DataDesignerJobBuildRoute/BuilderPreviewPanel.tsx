// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex } from '@nvidia/foundations-react-core';
import type { PreviewRecord } from '@studio/components/NewDataDesignerJobForm/previewApi';
import { PreviewResults } from '@studio/components/NewDataDesignerJobForm/PreviewResults';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { type FC, memo } from 'react';

export interface BuilderPreviewPanelProps {
  records: PreviewRecord[] | undefined;
  columnOrder: readonly string[];
  isPreviewing: boolean;
  error: string | undefined;
  isOpen: boolean;
  onToggle: () => void;
}

const previewSummary = (
  records: PreviewRecord[] | undefined,
  isPreviewing: boolean,
  error: string | undefined
): string => {
  if (records?.length)
    return `Preview · ${records.length} ${records.length === 1 ? 'row' : 'rows'}`;
  if (isPreviewing) return 'Preview · generating';
  return error ? 'Preview · failed' : 'Preview · no rows';
};

export const BuilderPreviewPanel: FC<BuilderPreviewPanelProps> = memo(function BuilderPreviewPanel({
  records,
  columnOrder,
  isPreviewing,
  error,
  isOpen,
  onToggle,
}) {
  return (
    <div className="flex max-h-[50vh] shrink-0 flex-col border-t border-base px-density-2xl">
      <Button
        kind="tertiary"
        color="neutral"
        size="small"
        className="self-start py-density-sm"
        aria-expanded={isOpen}
        onClick={onToggle}
      >
        <Flex align="center" gap="density-xs">
          {isOpen ? <ChevronDown size={14} aria-hidden /> : <ChevronRight size={14} aria-hidden />}
          {previewSummary(records, isPreviewing, error)}
        </Flex>
      </Button>

      {isOpen ? (
        <div className="flex min-h-0 flex-1 flex-col pb-density-lg">
          <PreviewResults
            records={records}
            columnOrder={columnOrder}
            isPreviewing={isPreviewing}
            error={error}
          />
        </div>
      ) : null}
    </div>
  );
});
