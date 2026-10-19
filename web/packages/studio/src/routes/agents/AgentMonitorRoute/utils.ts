// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import {
  RunSummary,
  TelemetrySpan,
  agentFromPath,
  parseSpans,
  reduceSpansToRuns,
  runIdFromPath,
} from '@studio/routes/agents/AgentMonitorRoute/telemetry';

export const FILESET_NAME = 'nemo-agent-telemetry';
export const MAX_FILES = 250;
export const DOWNLOAD_CONCURRENCY = 6;

export const isNotFoundError = (err: unknown): boolean => {
  const e = err as { response?: { status?: number }; status?: number };
  return e?.response?.status === 404 || e?.status === 404;
};

export interface FetchResult {
  runs: RunSummary[];
  fileCount: number;
}

interface FilesetFile {
  path: string;
}

// Round-robin newest-first per agent. A plain path-desc sort lets a late-
// alphabet agent's old runs crowd out earlier agents' recent runs.
export const sampleNewestPerAgent = <T extends FilesetFile>(files: T[], limit: number): T[] => {
  const groups = new Map<string, T[]>();
  for (const f of files) {
    const key = agentFromPath(f.path) ?? '';
    const arr = groups.get(key);
    if (arr) arr.push(f);
    else groups.set(key, [f]);
  }
  for (const arr of groups.values()) {
    arr.sort((a, b) => (a.path < b.path ? 1 : a.path > b.path ? -1 : 0));
  }
  const out: T[] = [];
  for (let i = 0; out.length < limit; i++) {
    let progressed = false;
    for (const arr of groups.values()) {
      if (i < arr.length) {
        out.push(arr[i]);
        progressed = true;
        if (out.length >= limit) break;
      }
    }
    if (!progressed) break;
  }
  return out;
};

// Caller (the route) lists the fileset once via useFilesListFilesetFiles and
// passes the result in. Listing here too would double the API traffic and risk
// snapshot drift between the empty-state check and the run aggregation.
export const fetchTelemetryRuns = async (
  workspace: string,
  files: FilesetFile[],
  signal?: AbortSignal
): Promise<FetchResult> => {
  const jsonlFiles = files.filter((f) => f.path.endsWith('.jsonl'));
  if (jsonlFiles.length === 0) return { runs: [], fileCount: 0 };

  const limited = sampleNewestPerAgent(jsonlFiles, MAX_FILES);
  const spans: TelemetrySpan[] = [];
  let cursor = 0;
  const worker = async (): Promise<void> => {
    while (cursor < limited.length) {
      const file = limited[cursor++];
      const blob = await filesDownloadFile(workspace, FILESET_NAME, file.path, signal);
      if (!blob) continue;
      const text = await blob.text();
      spans.push(...parseSpans(text, agentFromPath(file.path), runIdFromPath(file.path)));
    }
  };
  const workerCount = Math.min(DOWNLOAD_CONCURRENCY, limited.length);
  await Promise.all(Array.from({ length: workerCount }, worker));

  return { runs: reduceSpansToRuns(spans), fileCount: limited.length };
};

export interface TokensByBucket {
  timestamps: Date[];
  promptTokens: number[];
  completionTokens: number[];
  bucketMs: number;
}

const MINUTE_MS = 60 * 1000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;

/**
 * Picks a bucket size so the chart has enough data points to be readable,
 * without collapsing everything into one bar when runs span a short window.
 */
const chooseBucketMs = (runs: RunSummary[]): number => {
  const times = runs.map((r) => r.startedAt.getTime()).filter(Number.isFinite);
  if (times.length < 2) return HOUR_MS;
  const spanMs = Math.max(...times) - Math.min(...times);
  if (spanMs < 2 * HOUR_MS) return 5 * MINUTE_MS;
  if (spanMs < DAY_MS) return HOUR_MS;
  if (spanMs < 7 * DAY_MS) return 6 * HOUR_MS;
  return DAY_MS;
};

export const bucketTokensByTime = (runs: RunSummary[]): TokensByBucket => {
  const bucketMs = chooseBucketMs(runs);
  const buckets = new Map<number, { prompt: number; completion: number }>();

  for (const run of runs) {
    const ts = run.startedAt.getTime();
    if (!Number.isFinite(ts)) continue;
    const bucketKey = Math.floor(ts / bucketMs) * bucketMs;
    const existing = buckets.get(bucketKey) ?? { prompt: 0, completion: 0 };
    existing.prompt += run.promptTokens;
    existing.completion += run.completionTokens;
    buckets.set(bucketKey, existing);
  }

  const sortedKeys = Array.from(buckets.keys()).sort((a, b) => a - b);
  return {
    timestamps: sortedKeys.map((k) => new Date(k)),
    promptTokens: sortedKeys.map((k) => buckets.get(k)?.prompt ?? 0),
    completionTokens: sortedKeys.map((k) => buckets.get(k)?.completion ?? 0),
    bucketMs,
  };
};

export interface MonitorSummary {
  totalRuns: number;
  avgPromptTokens: number;
  avgCompletionTokens: number;
  totalToolCalls: number;
  topModel: string;
  topModelCount: number;
  uniqueModels: number;
  uniqueAgents: number;
}

export const summarizeRuns = (runs: RunSummary[]): MonitorSummary => {
  const totalRuns = runs.length;
  let promptSum = 0;
  let completionSum = 0;
  let toolCallSum = 0;
  const modelCounts = new Map<string, number>();
  const agents = new Set<string>();

  for (const run of runs) {
    promptSum += run.promptTokens;
    completionSum += run.completionTokens;
    toolCallSum += run.toolCalls;
    if (run.model) {
      modelCounts.set(run.model, (modelCounts.get(run.model) ?? 0) + 1);
    }
    if (run.agent) agents.add(run.agent);
  }

  let topModel = '—';
  let topModelCount = 0;
  for (const [model, count] of modelCounts) {
    if (count > topModelCount) {
      topModel = model;
      topModelCount = count;
    }
  }

  return {
    totalRuns,
    avgPromptTokens: totalRuns ? Math.round(promptSum / totalRuns) : 0,
    avgCompletionTokens: totalRuns ? Math.round(completionSum / totalRuns) : 0,
    totalToolCalls: toolCallSum,
    topModel,
    topModelCount,
    uniqueModels: modelCounts.size,
    uniqueAgents: agents.size,
  };
};
