// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogEditor } from '@nemo/common/src/components/LogViewer/LogEditor';
import type { NotifyFn } from '@nemo/common/src/providers/toast/types';
import { useNotify } from '@nemo/common/src/providers/toast/useNotify';
import { getErrorMessage } from '@nemo/common/src/utils/error';
import { triggerDownload } from '@nemo/common/src/utils/file';
import { formatLogs, formatLogsForDisplay, type LogPaging } from '@nemo/common/src/utils/logs';
import { ArrowUp, CopyGeneric, Download, Paragraph } from '@nemo/icons/icons';
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
import { FC, useState } from 'react';

import '@nemo/common/src/components/LogViewer/styles.css';

const DEFAULT_ROW_COUNT = 30;

interface LogViewerProps {
  logs: HelixJobLog[];
  isLoading?: boolean;
  downloadFilename?: string;
  rows?: number;
  fillHeight?: boolean;
  emptyMessage?: string;
  /** Reaches lines not passed in `logs`: older pages behind "Load previous logs", and the
   *  whole log for download. Without it, both stop at `logs`. */
  paging?: LogPaging;
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
  paging,
  onNotify,
}) => {
  const [showAllLogs, setShowAllLogs] = useState(fillHeight);
  const [wrapLines, setWrapLines] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const tailLogs = logs?.slice(-rows) || [];
  const displayedLogs = showAllLogs ? logs : tailLogs;
  // Copy what's on screen; download keeps the raw messages.
  const logText = formatLogsForDisplay(displayedLogs);
  const hasMoreLogs = logs && logs.length > rows;
  const hasHiddenLogs = !showAllLogs && hasMoreLogs;
  const isFetchingOlderLogs = !!paging?.isFetchingOlderLogs;

  const notify = useNotify(onNotify);

  const handleLoadPrevious = () => {
    setShowAllLogs(true);
    // Reveal the lines already here before asking the server for older ones.
    if (hasHiddenLogs || !paging) {
      return;
    }
    paging
      .fetchOlderLogs()
      .catch((error: unknown) =>
        notify(`Failed to load previous logs: ${getErrorMessage(error)}`, 'error')
      );
  };

  const handleDownload = async () => {
    if (!downloadFilename) {
      return;
    }
    // Without older lines on the server, what's loaded is already the whole log.
    if (!paging?.hasOlderLogs) {
      triggerDownload(formatLogs(logs), downloadFilename);
      return;
    }
    setIsDownloading(true);
    try {
      triggerDownload(formatLogs(await paging.fetchAllLogs()), downloadFilename);
    } catch (error) {
      notify(`Failed to download logs: ${getErrorMessage(error)}`, 'error');
    } finally {
      setIsDownloading(false);
    }
  };

  if (isLoading) {
    return (
      <Flex align="center" justify="center" className="h-full min-h-32 w-full">
        <Spinner size="medium" aria-label="Loading..." />
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
            <Text kind="mono/md">
              {displayedLogs.length.toLocaleString()}{' '}
              {!showAllLogs && hasMoreLogs && `of ${logs.length.toLocaleString()}`} lines
            </Text>
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
                  disabled={isDownloading}
                  onClick={() => void handleDownload()}
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
        {(hasHiddenLogs || paging?.hasOlderLogs) && (
          // Zero-height anchor so the fade starts at the editor's top edge.
          <Block className="relative z-10 h-0">
            <Block className="absolute inset-x-px top-px max-h-none rounded-t-md py-5 text-center bg-[linear-gradient(to_bottom,var(--nv-code-snippet-custom-background),transparent)]">
              <Tag
                color="gray"
                kind="solid"
                disabled={isFetchingOlderLogs}
                aria-busy={isFetchingOlderLogs}
                onClick={handleLoadPrevious}
              >
                <ArrowUp />
                {isFetchingOlderLogs ? 'Loading previous logs...' : 'Load previous logs'}
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
