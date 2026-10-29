// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogEditor } from '@nemo/common/src/components/LogViewer/LogEditor';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import type { NotifyFn } from '@nemo/common/src/providers/toast/types';
import { useNotify } from '@nemo/common/src/providers/toast/useNotify';
import { triggerDownload } from '@nemo/common/src/utils/file';
import {
  formatLogs,
  formatLogsForDisplay,
  type LogLoadProgress,
  type LogStatus,
} from '@nemo/common/src/utils/logs';
import { ArrowUp, CopyGeneric, Download, Paragraph, Record as RecordIcon } from '@nemo/icons/icons';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';
import {
  Block,
  Button,
  CodeSnippetActions,
  CodeSnippetCopyButton,
  CodeSnippetRoot,
  Flex,
  Spinner,
  Tag,
  Text,
} from '@nvidia/foundations-react-core';
import { FC, useEffect, useRef, useState } from 'react';

import '@nemo/common/src/components/LogViewer/styles.css';

const DEFAULT_ROW_COUNT = 30;

// The page walk reports after every page, which on a long job is several updates a
// second. Coalescing them keeps the aria-live region from queueing an announcement
// per page while still reading as live progress.
const PROGRESS_UPDATE_INTERVAL_MS = 1_000;

/** Emits `value` immediately, then at most once per `intervalMs`, always settling on
 *  the latest value. */
function useThrottledValue<T>(value: T, intervalMs: number): T {
  const [throttled, setThrottled] = useState(value);
  const lastEmitRef = useRef(0);

  useEffect(() => {
    if (value === throttled) return;
    const emit = () => {
      lastEmitRef.current = Date.now();
      setThrottled(value);
    };
    const waitMs = intervalMs - (Date.now() - lastEmitRef.current);
    if (waitMs <= 0) {
      emit();
      return;
    }
    const timer = setTimeout(emit, waitMs);
    return () => clearTimeout(timer);
  }, [value, throttled, intervalMs]);

  return throttled;
}

interface LogViewerProps {
  logs: HelixJobLog[];
  isLoading?: boolean;
  downloadFilename?: string;
  rows?: number;
  fillHeight?: boolean;
  emptyMessage?: string;
  /** Progress of the initial multi-page fetch, shown beneath the loading spinner.
   *  Ignored once logs are on screen. Callers that fetch in one shot can omit it. */
  loadProgress?: LogLoadProgress | null;
  /** Badges the header as Live or Ended. Omit when the caller can't tell. */
  logStatus?: LogStatus;
  /** Where the copy confirmation goes. Defaults to the surrounding ToastProvider; plugins pass `host.notifications.notify`. */
  onNotify?: NotifyFn;
}

export const LogViewer: FC<LogViewerProps> = ({
  logs,
  isLoading = false,
  downloadFilename,
  rows = DEFAULT_ROW_COUNT,
  fillHeight = false,
  emptyMessage = 'No logs available yet',
  loadProgress,
  logStatus,
  onNotify,
}) => {
  const [showAllLogs, setShowAllLogs] = useState(fillHeight);
  const [wrapLines, setWrapLines] = useState(false);
  const tailLogs = logs?.slice(-rows) || [];
  const displayedLogs = showAllLogs ? logs : tailLogs;
  // Copy what's on screen; download keeps the raw messages.
  const logText = formatLogsForDisplay(displayedLogs);
  const hasMoreLogs = logs && logs.length > rows;

  // Announce the wait straight away; the counts join it once the first page tells us
  // the denominator.
  const progressLabel =
    loadProgress && loadProgress.total > 0
      ? `Loading logs... ${loadProgress.loaded.toLocaleString()} of ${loadProgress.total.toLocaleString()} lines`
      : 'Loading logs...';
  const throttledProgressLabel = useThrottledValue(progressLabel, PROGRESS_UPDATE_INTERVAL_MS);

  const notify = useNotify(onNotify);

  const handleDownload = () => {
    if (downloadFilename) {
      triggerDownload(formatLogs(logs), downloadFilename);
    }
  };

  if (isLoading) {
    return (
      <Flex
        direction="column"
        align="center"
        justify="center"
        gap="density-sm"
        className="h-full min-h-32 w-full"
      >
        <Spinner size="medium" aria-label="Loading..." />
        {/* The fetch walks the log a page at a time, so on a large job this sits here
            for many round-trips. Mounted for every caller that opts into progress,
            so that the counts arriving with the first page are a mutation inside a
            live region already in the DOM — screen readers that only announce such
            mutations skip a region that appears already populated. */}
        {loadProgress !== undefined && (
          <Text
            kind="mono/md"
            className="text-subtle"
            aria-live="polite"
            data-testid="log-load-progress"
          >
            {throttledProgressLabel}
          </Text>
        )}
      </Flex>
    );
  }

  if (!logs || logs.length === 0) {
    return (
      <Flex align="center" justify="center" className="h-full min-h-32 w-full">
        <Block className="text-subtle">{emptyMessage}</Block>
      </Flex>
    );
  }

  return (
    <Block className="relative w-full min-w-0 max-w-full overflow-hidden h-full">
      <CodeSnippetRoot
        kind="block"
        collapsible={false}
        rows={fillHeight ? undefined : rows}
        className="log-viewer min-h-auto h-full"
      >
        <CodeSnippetActions>
          <Flex className="w-full" justify="between" wrap="wrap">
            <Flex gap="density-sm" align="center">
              <Text kind="mono/md">
                {displayedLogs.length.toLocaleString()}{' '}
                {!showAllLogs && hasMoreLogs && `of ${logs.length.toLocaleString()}`} lines
              </Text>
              {logStatus && (
                <StatusBadge
                  status={logStatus}
                  statusConfig={{
                    live: { label: 'Live', color: 'green', icon: RecordIcon },
                    ended: { label: 'Ended', color: 'gray' },
                  }}
                />
              )}
            </Flex>
            <Flex gap="0.25">
              <Button
                size="tiny"
                kind={wrapLines ? 'secondary' : 'tertiary'}
                title="Wrap lines"
                aria-label="Wrap lines"
                aria-pressed={wrapLines}
                onClick={() => setWrapLines((prev) => !prev)}
              >
                <Paragraph />
              </Button>
              {downloadFilename && (
                <Button
                  size="tiny"
                  kind="tertiary"
                  title="Download logs"
                  aria-label="Download logs"
                  onClick={handleDownload}
                >
                  <Download />
                </Button>
              )}
            </Flex>
          </Flex>
          <CodeSnippetCopyButton
            value={logText}
            title="Copy logs"
            aria-label="Copy logs"
            onClick={() => notify('Copied to clipboard!', 'success', { durationMs: 3000 })}
          >
            <CopyGeneric />
          </CodeSnippetCopyButton>
        </CodeSnippetActions>
        {!showAllLogs && hasMoreLogs && (
          // Zero-height anchor so the fade starts at the editor's top edge.
          <Block className="relative z-10 h-0">
            <Block className="absolute inset-x-px top-px max-h-none rounded-t-md py-5 text-center bg-[linear-gradient(to_bottom,var(--nv-code-snippet-custom-background),transparent)]">
              <Tag color="gray" kind="solid" onClick={() => setShowAllLogs(true)}>
                <ArrowUp />
                Load previous logs
              </Tag>
            </Block>
          </Block>
        )}
        <LogEditor
          logs={displayedLogs}
          resetKey={showAllLogs}
          wrapLines={wrapLines}
          fillHeight={fillHeight}
          // The unwrapped tail is sized to its rows, so it doesn't scroll.
          scrollLocked={!showAllLogs && !wrapLines}
        />
      </CodeSnippetRoot>
    </Block>
  );
};
