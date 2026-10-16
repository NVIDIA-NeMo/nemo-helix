// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useStickToBottom } from '@nemo/common/src/hooks/useStickToBottom';
import {
  Button,
  CodeSnippet,
  Flex,
  Spinner,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import {
  formatPreviewLogsForDisplay,
  type PreviewRecord,
} from '@studio/components/NewDataDesignerJobForm/previewApi';
import { PreviewResults } from '@studio/components/NewDataDesignerJobForm/PreviewResults';
import { useResizableHeight } from '@studio/routes/DataDesignerJobBuildRoute/useResizableHeight';
import cn from 'classnames';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { type FC, memo, useMemo, useState } from 'react';

type PreviewTab = 'rows' | 'logs';

const DEFAULT_PANEL_HEIGHT_PX = 320;
const MIN_PANEL_HEIGHT_PX = 160;
const MAX_PANEL_VIEWPORT_RATIO = 0.8;
const LOGS_CHROME_PX = 130;

export interface BuilderPreviewPanelProps {
  records: PreviewRecord[] | undefined;
  columnOrder: readonly string[];
  logs: string;
  status: string | undefined;
  isPreviewing: boolean;
  error: string | undefined;
  isExpanded: boolean;
  onToggleExpanded: () => void;
  onStop: () => void;
}

export const BuilderPreviewPanel: FC<BuilderPreviewPanelProps> = memo(function BuilderPreviewPanel({
  records,
  columnOrder,
  logs,
  status,
  isPreviewing,
  error,
  isExpanded,
  onToggleExpanded,
  onStop,
}) {
  const [tab, setTab] = useState<PreviewTab>('rows');
  const { ref: logsScrollRef } = useStickToBottom<HTMLDivElement>({
    enabled: isExpanded && tab === 'logs' && !!logs,
  });
  const displayedLogs = useMemo(
    () => (logs ? formatPreviewLogsForDisplay(logs) : 'No logs yet.'),
    [logs]
  );
  const resize = useResizableHeight({
    defaultHeight: DEFAULT_PANEL_HEIGHT_PX,
    minHeight: MIN_PANEL_HEIGHT_PX,
    maxViewportRatio: MAX_PANEL_VIEWPORT_RATIO,
  });

  return (
    <div
      // eslint-disable-next-line no-restricted-syntax
      style={isExpanded ? { height: resize.height } : undefined}
      className="relative flex shrink-0 flex-col border-t border-base px-density-2xl"
    >
      {isExpanded ? (
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
      ) : null}

      <TabsRoot
        value={tab}
        onValueChange={(value) => setTab(value as PreviewTab)}
        className={cn(
          'flex min-h-0 flex-1 flex-col pt-density-sm',
          isExpanded ? 'pb-density-lg' : 'pb-density-sm'
        )}
      >
        <Flex
          align="center"
          justify="between"
          gap="density-md"
          className={cn(isExpanded && 'shadow-[inset_0_-2px_0_0_var(--border-color-base)]')}
        >
          {isExpanded ? (
            <TabsList aria-label="Preview output" className="shadow-none!">
              <TabsTrigger value="rows">Preview Rows</TabsTrigger>
              <TabsTrigger value="logs">Preview Run Logs</TabsTrigger>
            </TabsList>
          ) : (
            <Flex align="center" gap="density-sm" className="h-10">
              <Text kind="label/bold/md">Preview Rows</Text>
              {isPreviewing ? (
                <Spinner
                  size="small"
                  aria-label="Generating preview rows"
                  className="[&>div:first-child]:h-4!"
                />
              ) : null}
            </Flex>
          )}
          <Button
            kind="tertiary"
            color="neutral"
            size="small"
            aria-label={isExpanded ? 'Collapse preview' : 'Expand preview'}
            onClick={onToggleExpanded}
          >
            {isExpanded ? (
              <ChevronDown size={16} aria-hidden />
            ) : (
              <ChevronUp size={16} aria-hidden />
            )}
          </Button>
        </Flex>

        {isExpanded ? (
          <>
            <TabsContent
              value="rows"
              className="min-h-0 flex-1 flex-col items-stretch p-0 pt-density-lg"
            >
              <PreviewResults
                records={records}
                columnOrder={columnOrder}
                isPreviewing={isPreviewing}
                status={status}
                error={error}
                onStop={onStop}
              />
            </TabsContent>

            <TabsContent
              value="logs"
              className="min-h-0 flex-1 flex-col items-stretch p-0 pt-density-lg"
            >
              <CodeSnippet
                value={displayedLogs}
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
          </>
        ) : null}
      </TabsRoot>
    </div>
  );
});
