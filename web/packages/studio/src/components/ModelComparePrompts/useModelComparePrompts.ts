// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SubmitUploadType } from '@nemo/common/src/components/UploadModal/types';
import { useChatCompletion } from '@nemo/common/src/hooks/useChatCompletion';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { logger } from '@nemo/common/src/utils/logger';
import { type FileSampleMethod } from '@nemo/common/src/utils/sampleTextLines';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { SAMPLE_DATASETS } from '@studio/components/chat/sampleDatasets';
import {
  DEFAULT_SAMPLE_SIZE,
  FILESET_PICKER_VALUE,
  INFERENCE_BATCH_SIZE,
  UPLOADED_FILE_VALUE,
} from '@studio/components/ModelComparePrompts/constants';
import {
  buildPromptRowsFromParsedRows,
  parseUploadedFile,
} from '@studio/components/ModelComparePrompts/helpers';
import type {
  ExpandedCellState,
  ModelComparePromptsProps,
  PromptRow,
  ResponseResult,
  ResponseStats,
} from '@studio/components/ModelComparePrompts/types';
import type { DatasetInputFileResult } from '@studio/types/dataset';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type UseModelComparePromptsArgs = Pick<
  ModelComparePromptsProps,
  'workspace' | 'models' | 'onReadyChange' | 'agentName'
>;

export function useModelComparePrompts({
  workspace,
  models,
  onReadyChange,
  agentName,
}: UseModelComparePromptsArgs) {
  const [fileResult, setFileResult] = useState<DatasetInputFileResult | null>(null);
  const [promptRows, setPromptRows] = useState<PromptRow[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [sampleSize, setSampleSize] = useState<number>(DEFAULT_SAMPLE_SIZE);
  const [sampleMethod, setSampleMethod] = useState<FileSampleMethod>('random');
  const [expandedCell, setExpandedCell] = useState<ExpandedCellState | null>(null);
  const [pickerValue, setPickerValue] = useState<string | undefined>(undefined);
  // Bumped to remount the dataset Select after the "Select from dataset file..."
  // sentinel is chosen, so the action can be retriggered (re-selecting the same
  // option otherwise fires no change event).
  const [pickerSelectKey, setPickerSelectKey] = useState(0);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [isFilesetPickerOpen, setIsFilesetPickerOpen] = useState(false);
  // True when the loaded file's prompt column was auto-detected. In that case
  // we hide the manual column picker; we only surface it when detection failed.
  const [promptKeyAutoDetected, setPromptKeyAutoDetected] = useState(false);
  const { mutateAsync: createCompletion } = useChatCompletion();

  // Monotonic run id. Incremented on invalidation; guards stale writeCell calls.
  const runIdRef = useRef(0);
  // AbortController for the active run; aborted when a new run starts,
  // dataset/sampling changes, or the component unmounts.
  const runAbortRef = useRef<AbortController | null>(null);

  const rowCount = fileResult?.rowCount ?? 0;

  const handleFileChange = useCallback((result: DatasetInputFileResult | null) => {
    runIdRef.current += 1;
    runAbortRef.current?.abort();
    setFileResult(result);
    setPromptRows([]);
    setPromptKeyAutoDetected(result?.keyMapping.promptKey != null);
    if (result) {
      setSampleSize(Math.min(DEFAULT_SAMPLE_SIZE, result.rowCount || DEFAULT_SAMPLE_SIZE));
    }
  }, []);

  // Override the auto-detected prompt column. Updating `keyMapping.promptKey`
  // triggers the row-rebuild effect below; fresh rows clear stale responses.
  const handlePromptKeyChange = useCallback((key: string) => {
    setFileResult((prev) =>
      prev ? { ...prev, keyMapping: { ...prev.keyMapping, promptKey: key } } : prev
    );
  }, []);

  /**
   * Clear cached inference responses. If `columnId` is provided, only that
   * column's responses are cleared (e.g. when a new model is picked for the
   * column). If omitted, all responses across all columns are cleared
   * (e.g. on Run, or when picking new random prompts).
   */
  const clearResponses = useCallback((columnId?: number) => {
    setPromptRows((prev) =>
      prev.map((row) => {
        if (columnId === undefined) {
          return { ...row, responses: {} };
        }
        const next = { ...row.responses };
        delete next[columnId];
        return { ...row, responses: next };
      })
    );
  }, []);

  const runInference = useCallback(async () => {
    const activeModels = models
      .map((m) => {
        if (!m.modelURN) return null;
        const { workspace: modelWorkspace, name } = getPartsFromReference(m.modelURN);
        return { id: m.id, modelWorkspace, name };
      })
      .filter((m): m is { id: number; modelWorkspace: string; name: string } => m !== null);

    if (activeModels.length === 0 || promptRows.length === 0) return;

    // Snapshot inputs at start of run; any later change invalidates this run.
    const snapshotPromptRows = promptRows;
    const snapshotActiveModels = activeModels;
    runIdRef.current += 1;
    const myRunId = runIdRef.current;

    runAbortRef.current?.abort();
    const runController = new AbortController();
    runAbortRef.current = runController;

    setIsRunning(true);
    clearResponses();

    // Writes a single cell's result, but only if this run is still current.
    const writeCell = (sourceIndex: number, modelId: number, result: ResponseResult | null) => {
      if (runIdRef.current !== myRunId) return;
      setPromptRows((prev) =>
        prev.map((row) =>
          row.sourceIndex === sourceIndex
            ? { ...row, responses: { ...row.responses, [modelId]: result } }
            : row
        )
      );
    };

    // Build task factories (not yet fired). Each one updates its own cell as
    // soon as it resolves so results stream in.
    const taskFactories: Array<() => Promise<void>> = [];
    snapshotActiveModels.forEach((model) => {
      snapshotPromptRows.forEach((row) => {
        taskFactories.push(() => {
          const startTime = performance.now();
          return createCompletion({
            model: model.name,
            workspace: model.modelWorkspace || workspace,
            messages: [{ role: 'user', content: row.prompt }],
            stream: false,
            signal: runController.signal,
          })
            .then((result) => {
              const totalMs = performance.now() - startTime;
              const content =
                result && 'choices' in result
                  ? (result.choices[0]?.message?.content ?? null)
                  : null;
              if (content === null) {
                writeCell(row.sourceIndex, model.id, null);
                return;
              }
              const usage = result && 'usage' in result ? result.usage : undefined;
              // Fallback estimate: ~4 chars per token. Good enough for the badge when
              // the gateway elides usage stats.
              const completionTokens =
                usage?.completion_tokens ?? Math.max(1, Math.round(content.length / 4));
              const tokensPerSec = totalMs > 0 ? completionTokens / (totalMs / 1000) : 0;
              writeCell(row.sourceIndex, model.id, {
                text: content,
                stats: { totalMs, completionTokens, tokensPerSec },
              });
            })
            .catch((error) => {
              logger.error('Inference request failed', error);
              writeCell(row.sourceIndex, model.id, null);
            });
        });
      });
    });

    // Run tasks in capped-size batches so we don't flood the gateway.
    try {
      for (let i = 0; i < taskFactories.length; i += INFERENCE_BATCH_SIZE) {
        if (runController.signal.aborted) break;
        const batch = taskFactories.slice(i, i + INFERENCE_BATCH_SIZE).map((fn) => fn());
        await Promise.allSettled(batch);
      }
    } finally {
      if (runAbortRef.current === runController) {
        runAbortRef.current = null;
        setIsRunning(false);
      }
    }
  }, [models, promptRows, workspace, createCompletion, clearResponses]);

  // Cancel an in-flight run without clearing results. Bumping the run id makes
  // any writes from aborted (rejected) requests no-op, so completed cells keep
  // their results and still-pending cells stay blank. A later Run clears all.
  const cancelRun = useCallback(() => {
    runIdRef.current += 1;
    runAbortRef.current?.abort();
    runAbortRef.current = null;
    setIsRunning(false);
  }, []);

  const hasPromptKey = fileResult?.keyMapping.promptKey != null;
  const hasAssignedModel = models.some((m) => m.modelURN !== null);
  const hasPrompts = promptRows.length > 0;

  /**
   * Per-column averages across all completed responses. `tokensPerSec` is
   * weighted (sum tokens / sum seconds) rather than a mean-of-means so short
   * responses don't over-influence the rate. Returns null for columns with
   * zero completed responses so the footer can render an em-dash.
   */
  const averagesByModelId = useMemo(() => {
    const result: Record<number, (ResponseStats & { count: number }) | null> = {};
    models.forEach((m) => {
      let totalMs = 0;
      let totalTokens = 0;
      let count = 0;
      promptRows.forEach((row) => {
        const r = row.responses[m.id];
        if (!r) return;
        totalMs += r.stats.totalMs;
        totalTokens += r.stats.completionTokens;
        count += 1;
      });
      if (count === 0) {
        result[m.id] = null;
        return;
      }
      result[m.id] = {
        totalMs: totalMs / count,
        completionTokens: totalTokens / count,
        tokensPerSec: totalMs > 0 ? totalTokens / (totalMs / 1000) : 0,
        count,
      };
    });
    return result;
  }, [models, promptRows]);

  const anyAverages = Object.values(averagesByModelId).some((a) => a !== null);

  // Notify parent when readiness changes. "Ready" means the table is active
  // (file is loaded and has a valid prompt key mapped).
  const isReady = !!fileResult && hasPromptKey;
  useEffect(() => {
    onReadyChange?.(isReady);
  }, [isReady, onReadyChange]);

  // Abort any active run on unmount (e.g. tab switch, navigation).
  useEffect(() => {
    return () => {
      runAbortRef.current?.abort();
    };
  }, []);

  // Drive the prompt table from parsed preview rows + sampling controls (no separate file preview).
  useEffect(() => {
    if (!fileResult?.keyMapping.promptKey || !fileResult.parsedRows?.length) return;

    runIdRef.current += 1;
    runAbortRef.current?.abort();
    setPromptRows(buildPromptRowsFromParsedRows(fileResult, sampleSize, sampleMethod));
  }, [fileResult, sampleSize, sampleMethod]);

  // Auto-select the agent's matching sample when the user lands on Run Prompts
  // via the agent overlay. Tracks the last-auto-selected agent in a ref so we
  // don't re-fire after the user clears the picker or picks a different file.
  const autoSelectedAgentRef = useRef<string | null>(null);
  useEffect(() => {
    if (!agentName) {
      autoSelectedAgentRef.current = null;
      return;
    }
    if (autoSelectedAgentRef.current === agentName) return;
    const match = SAMPLE_DATASETS.find((s) => s.id === agentName);
    if (!match) return;
    autoSelectedAgentRef.current = agentName;
    setPickerValue(match.id);
    setUploadedFileName(null);
    setParseError(null);
    handleFileChange(match.build());
    // We intentionally re-run only on `agentName` change. Including
    // `handleFileChange` (or the various setters) would re-fire this effect
    // every time the parent re-renders and produce a seed loop — the agentRef
    // guard above would still no-op the work, but the effect would still run
    // and we want the dependencies to read true.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentName]);

  /**
   * Single picker handler. Three branches:
   *  - sample id → synthesize the result via `sample.build()` (in-memory)
   *  - upload sentinel → click the hidden native file input
   *  - uploaded sentinel → no-op (it's the displayed value after a successful upload)
   */
  const handleDatasetSelect = useCallback(
    (value: string) => {
      if (!value) return;
      if (value === UPLOADED_FILE_VALUE) return;
      if (value === FILESET_PICKER_VALUE) {
        setIsFilesetPickerOpen(true);
        setPickerSelectKey((k) => k + 1);
        return;
      }
      const sample = SAMPLE_DATASETS.find((s) => s.id === value);
      if (!sample) return;
      setParseError(null);
      setUploadedFileName(null);
      setPickerValue(value);
      handleFileChange(sample.build());
    },
    [handleFileChange]
  );

  const handleFilesetPickerSubmit = useCallback(
    async (data: SubmitUploadType) => {
      if (data.type !== 'dataset') return;
      setIsFilesetPickerOpen(false);
      setParseError(null);
      try {
        // `data.url` is a `fileset://` URI, not an HTTP URL — download via the
        // SDK using the dataset's workspace/name and the file path.
        const response = await filesDownloadFile(
          data.dataset.workspace,
          data.dataset.name,
          data.path
        );
        if (!response) {
          setParseError('Failed to download file');
          return;
        }
        const text = await response.text();
        const filename = data.path.split('/').pop() ?? 'dataset.json';
        const file = new File([text], filename);
        const result = await parseUploadedFile(file);
        if ('error' in result) {
          setParseError(result.error);
          return;
        }
        setUploadedFileName(`${data.dataset.name}/${data.path}`);
        setPickerValue(UPLOADED_FILE_VALUE);
        handleFileChange(result);
      } catch (err) {
        setParseError(err instanceof Error ? err.message : 'Failed to load file');
      }
    },
    [handleFileChange]
  );

  const datasetItems = useMemo(() => {
    const items: { value: string; children: string }[] = SAMPLE_DATASETS.map((s) => ({
      value: s.id,
      children: s.label,
    }));
    if (uploadedFileName) {
      items.push({ value: UPLOADED_FILE_VALUE, children: uploadedFileName });
    }
    items.push({ value: FILESET_PICKER_VALUE, children: 'Select from dataset file...' });
    return items;
  }, [uploadedFileName]);

  return {
    fileResult,
    promptRows,
    isRunning,
    sampleSize,
    setSampleSize,
    sampleMethod,
    setSampleMethod,
    expandedCell,
    setExpandedCell,
    pickerValue,
    pickerSelectKey,
    parseError,
    isFilesetPickerOpen,
    setIsFilesetPickerOpen,
    promptKeyAutoDetected,
    rowCount,
    handlePromptKeyChange,
    clearResponses,
    runInference,
    cancelRun,
    hasAssignedModel,
    hasPrompts,
    averagesByModelId,
    anyAverages,
    handleDatasetSelect,
    handleFilesetPickerSubmit,
    datasetItems,
  };
}
