// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useStickToBottom } from '@nemo/common/src/hooks/useStickToBottom';
import {
  Button,
  CodeSnippet,
  Flex,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
} from '@nvidia/foundations-react-core';
import {
  formatPreviewLogsForDisplay,
  type PreviewRecord,
} from '@studio/components/NewDataDesignerJobForm/previewApi';
import { PreviewResults } from '@studio/components/NewDataDesignerJobForm/PreviewResults';
import { useResizableHeight } from '@studio/routes/DataDesignerJobBuildRoute/useResizableHeight';
import cn from 'classnames';
import { X } from 'lucide-react';
import { type FC, memo, useState } from 'react';

type PreviewTab = 'rows' | 'logs';

const DEFAULT_PANEL_HEIGHT_PX = 320;
const MIN_PANEL_HEIGHT_PX = 160;
const MAX_PANEL_VIEWPORT_RATIO = 0.8;
const LOGS_CHROME_PX = 130;

export interface BuilderPreviewPanelProps {
  records: PreviewRecord[] | undefined;
  columnOrder: readonly string[];
  logs: string;
  isPreviewing: boolean;
  error: string | undefined;
  onClose: () => void;
}

export const BuilderPreviewPanel: FC<BuilderPreviewPanelProps> = memo(function BuilderPreviewPanel({
  records,
  columnOrder,
  logs,
  isPreviewing,
  error,
  onClose,
}) {
  const [tab, setTab] = useState<PreviewTab>('rows');
  const { ref: logsScrollRef } = useStickToBottom<HTMLDivElement>({
    enabled: tab === 'logs' && !!logs,
  });
  const resize = useResizableHeight({
    defaultHeight: DEFAULT_PANEL_HEIGHT_PX,
    minHeight: MIN_PANEL_HEIGHT_PX,
    maxViewportRatio: MAX_PANEL_VIEWPORT_RATIO,
  });

  return (
    <div
      // eslint-disable-next-line no-restricted-syntax
      style={{ height: resize.height }}
      className="relative flex shrink-0 flex-col border-t border-base px-density-2xl"
    >
      <div
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize preview panel"
        aria-valuemin={resize.minHeight}
        aria-valuemax={resize.maxHeight}
        aria-valuenow={resize.height}
        tabIndex={0}
        className="group absolute inset-x-0 top-[-5px] z-10 flex h-[10px] cursor-row-resize items-center justify-center focus-visible:outline-none"
        onMouseDown={resize.onMouseDown}
        onKeyDown={resize.onKeyDown}
      >
        <span
          className={cn(
            'h-1 w-10 rounded-full bg-border-base transition-colors group-hover:bg-border-strong group-focus-visible:bg-border-brand',
            resize.isDragging && 'bg-border-brand'
          )}
        />
      </div>

      <TabsRoot
        value={tab}
        onValueChange={(value) => setTab(value as PreviewTab)}
        className="flex min-h-0 flex-1 flex-col pb-density-lg pt-density-sm"
      >
        <Flex align="center" justify="between" gap="density-md">
          <TabsList aria-label="Preview output">
            <TabsTrigger value="rows">Preview Rows</TabsTrigger>
            <TabsTrigger value="logs">Preview Run Logs</TabsTrigger>
          </TabsList>
          <Button
            kind="tertiary"
            color="neutral"
            size="small"
            aria-label="Close preview"
            onClick={onClose}
          >
            <X size={16} aria-hidden />
          </Button>
        </Flex>

        <TabsContent
          value="rows"
          className="min-h-0 flex-1 flex-col items-stretch p-0 pt-density-lg"
        >
          <PreviewResults
            records={records}
            columnOrder={columnOrder}
            isPreviewing={isPreviewing}
            error={error}
          />
        </TabsContent>

        <TabsContent
          value="logs"
          className="min-h-0 flex-1 flex-col items-stretch p-0 pt-density-lg"
        >
          <CodeSnippet
            value={logs ? formatPreviewLogsForDisplay(logs) : 'No logs yet.'}
            language="json"
            kind="block"
            attributes={{
              CodeSnippetCode: {
                ref: logsScrollRef,
                style: { maxHeight: Math.max(resize.height - LOGS_CHROME_PX, 0) },
              },
            }}
          />
        </TabsContent>
      </TabsRoot>
    </div>
  );
});
