// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isAbortError } from '@nemo/common/src/components/AssistantChat/completionUtils';
import type { PreviewRequest } from '@nemo/sdk/generated/anonymizer/schema';
import {
  streamAnonymizerPreview,
  type PreviewFrame,
} from '@studio/routes/AnonymizerBuilderRoute/previewApi';
import { startTransition, useCallback, useEffect, useRef, useState } from 'react';

const DEFAULT_TEXT_COLUMN = 'text';

interface PreviewResult {
  readonly records: Record<string, unknown>[];
  readonly textColumn: string;
  readonly failedRecords: Record<string, unknown>[];
}

const EMPTY_RESULT: PreviewResult = {
  records: [],
  textColumn: DEFAULT_TEXT_COLUMN,
  failedRecords: [],
};

export interface UseAnonymizerPreviewOptions {
  readonly workspace: string;
  readonly accessToken: string | undefined;
  /** Returns the request to preview, or undefined when the form isn't ready. */
  readonly getRequest: () => Promise<PreviewRequest | undefined>;
}

export interface UseAnonymizerPreview {
  readonly result: PreviewResult;
  readonly logs: readonly string[];
  readonly isPreviewing: boolean;
  /** True while the first-use PII detector weights are downloading. */
  readonly isDownloadingModel: boolean;
  /** Human-readable detail for the downloading state, when known. */
  readonly downloadMessage: string | undefined;
  readonly error: string | undefined;
  readonly hasRun: boolean;
  readonly wasStopped: boolean;
  readonly runPreview: () => Promise<void>;
  /** The server unwinds on disconnect, but an in-flight model call still finishes. */
  readonly stopPreview: () => void;
}

export const useAnonymizerPreview = ({
  workspace,
  accessToken,
  getRequest,
}: UseAnonymizerPreviewOptions): UseAnonymizerPreview => {
  const [result, setResult] = useState<PreviewResult>(EMPTY_RESULT);
  const [logs, setLogs] = useState<string[]>([]);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isDownloadingModel, setIsDownloadingModel] = useState(false);
  const [downloadMessage, setDownloadMessage] = useState<string | undefined>(undefined);
  const [error, setError] = useState<string | undefined>(undefined);
  const [hasRun, setHasRun] = useState(false);
  const [wasStopped, setWasStopped] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const runIdRef = useRef(0);

  useEffect(() => () => abortRef.current?.abort(), []);

  const stopPreview = useCallback(() => {
    if (!abortRef.current) return;
    setWasStopped(true);
    abortRef.current.abort();
  }, []);

  const runPreview = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const runId = ++runIdRef.current;
    const isCurrent = (): boolean => runIdRef.current === runId;

    const onFrame = (frame: PreviewFrame) => {
      if (!isCurrent()) return;
      switch (frame.kind) {
        case 'log':
          // One frame per read is too far apart for React to batch; keeps Stop responsive.
          startTransition(() => {
            if (!isCurrent()) return;
            setLogs((prev) => [...prev, frame.message]);
          });
          break;
        case 'model_download':
          setIsDownloadingModel(frame.status === 'started');
          setDownloadMessage(frame.status === 'started' ? frame.message : undefined);
          break;
        case 'trace_dataset':
          setResult((prev) => ({
            ...prev,
            records: frame.records,
            textColumn: frame.original_text_column ?? prev.textColumn,
          }));
          break;
        case 'failed_records':
          setResult((prev) => ({ ...prev, failedRecords: frame.records }));
          break;
        case 'error':
          setError(frame.message);
          break;
        default:
          break;
      }
    };

    try {
      const request = await getRequest();
      if (!isCurrent()) return;
      if (!request) {
        abortRef.current = null;
        return;
      }

      setResult(EMPTY_RESULT);
      setLogs([]);
      setIsDownloadingModel(false);
      setDownloadMessage(undefined);
      setError(undefined);
      setHasRun(true);
      setWasStopped(false);
      setIsPreviewing(true);

      await streamAnonymizerPreview(workspace, request, accessToken, controller.signal, onFrame);
    } catch (err) {
      if (isAbortError(err) || !isCurrent()) return;
      setError((err instanceof Error && err.message) || 'The preview run failed.');
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setIsPreviewing(false);
        setIsDownloadingModel(false);
      }
    }
  }, [workspace, accessToken, getRequest]);

  return {
    result,
    logs,
    isPreviewing,
    isDownloadingModel,
    downloadMessage,
    error,
    hasRun,
    wasStopped,
    runPreview,
    stopPreview,
  };
};
