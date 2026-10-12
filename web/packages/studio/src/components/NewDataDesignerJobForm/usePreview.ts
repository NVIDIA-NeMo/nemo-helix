// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/utils/error';
import type { DataDesignerConfig } from '@nemo/sdk/generated/data-designer/schema';
import {
  isAbortError,
  type PreviewRecord,
  streamPreview,
} from '@studio/components/NewDataDesignerJobForm/previewApi';
import { useCallback, useRef, useState } from 'react';

const PREVIEW_PATH_TEMPLATE = '/apis/data-designer/v2/workspaces/:workspace/preview';

function buildPreviewPath(workspace: string): string {
  return PREVIEW_PATH_TEMPLATE.replace(':workspace', encodeURIComponent(workspace));
}

export interface UsePreviewOptions {
  workspace: string;
  accessToken: string | undefined;
  getCurrentConfig: () => DataDesignerConfig | undefined;
}

export interface UsePreviewResult {
  previewLogs: string;
  /** Undefined until the stream delivers its dataset frame. */
  previewRecords: PreviewRecord[] | undefined;
  /** Column names of the previewed config, in config order. */
  previewColumnOrder: string[];
  previewError: string | undefined;
  hasPreviewRun: boolean;
  isPreviewing: boolean;
  runPreview: () => Promise<void>;
  /** Aborts the in-flight preview. A no-op when nothing is running. */
  stopPreview: () => void;
}

const configColumnNames = (config: DataDesignerConfig): string[] =>
  Array.isArray(config.columns) ? config.columns.map((column) => column.name) : [];

/**
 * Encapsulates preview state and streaming: runs the data designer preview endpoint,
 * appends log lines to previewLogs and collects the generated records. Supports abort on re-run.
 */
export function usePreview({
  workspace,
  accessToken,
  getCurrentConfig,
}: UsePreviewOptions): UsePreviewResult {
  const [previewLogs, setPreviewLogs] = useState('');
  const [previewRecords, setPreviewRecords] = useState<PreviewRecord[] | undefined>(undefined);
  const [previewColumnOrder, setPreviewColumnOrder] = useState<string[]>([]);
  const [previewError, setPreviewError] = useState<string | undefined>(undefined);
  const [hasPreviewRun, setHasPreviewRun] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  /** Identifies the newest run, so a superseded one can't clear state that isn't its own. */
  const runIdRef = useRef(0);

  const stopPreview = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const appendLogLine = useCallback((line: string) => {
    setPreviewLogs((prev) => (prev ? `${prev}\n${line}` : line));
  }, []);

  const runPreview = useCallback(async () => {
    const failBeforeStreaming = (message: string) => {
      setPreviewLogs(message);
      setPreviewRecords(undefined);
      setPreviewError(message);
      setHasPreviewRun(true);
    };

    let config: DataDesignerConfig | undefined;
    try {
      config = getCurrentConfig();
    } catch (err) {
      failBeforeStreaming(
        getErrorMessage(err, 'Could not build a preview config from the recipe.')
      );
      return;
    }
    if (!config) {
      failBeforeStreaming('Generate a job spec first, then run Preview.');
      return;
    }

    setPreviewLogs('');
    setPreviewRecords(undefined);
    setPreviewColumnOrder(configColumnNames(config));
    setPreviewError(undefined);
    setHasPreviewRun(true);
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const signal = abortRef.current.signal;
    const runId = ++runIdRef.current;
    const isCurrent = () => runIdRef.current === runId;
    setIsPreviewing(true);

    try {
      await streamPreview(
        buildPreviewPath(workspace),
        { config, num_records: 10 },
        accessToken,
        signal,
        {
          onLog: appendLogLine,
          onRecords: (records) => {
            if (isCurrent()) setPreviewRecords(records);
          },
          onError: (message) => {
            if (isCurrent()) setPreviewError(message);
          },
        }
      );
    } catch (err) {
      if (isAbortError(err)) {
        if (isCurrent()) appendLogLine('Preview stopped.');
        return;
      }
      const message = getErrorMessage(err, 'Preview request failed.');
      appendLogLine(message);
      if (isCurrent()) setPreviewError(message);
    } finally {
      if (isCurrent()) {
        setIsPreviewing(false);
        abortRef.current = null;
      }
    }
  }, [workspace, accessToken, getCurrentConfig, appendLogLine]);

  return {
    previewLogs,
    previewRecords,
    previewColumnOrder,
    previewError,
    hasPreviewRun,
    isPreviewing,
    runPreview,
    stopPreview,
  };
}
