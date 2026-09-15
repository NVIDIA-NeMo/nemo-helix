// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { customFetch } from '@nemo/sdk/generated/fetchers/platform';
import {
  filesListFilesetFiles,
  getFilesDownloadFileQueryKey,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { queryOptions } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import { parquetReadObjects } from 'hyparquet';

/**
 * Upper bounds on what a template may actually pull.
 *
 * Checked per shard inside the read loop rather than across every match, because the loop
 * stops as soon as the row quota is met — a repo re-sharded into hundreds of files is fine
 * so long as the rows we need come from the first one or two. Each check still runs before
 * its shard downloads, using the size the listing already reported.
 *
 * 64MB is ~29x the only shipped dataset: `xu3kev/BIRD-SQL-data-train` is a single 2.2MB
 * Parquet file. (Its rows total ~46MB as raw JSON, but Parquet dictionary-encodes the
 * CREATE TABLE schema each row repeats, so the file on the Hub is far smaller.) The cap is
 * a hard failure, so it is set well clear of a plausible upstream re-upload.
 */
const MAX_DATASET_SHARDS = 8;
const MAX_DATASET_BYTES = 64 * 1024 * 1024;

const MAX_RETRIES = 2;
const RETRY_BASE_DELAY_MS = 300;

/** A request that failed in a way worth retrying: network blip, rate limit, or 5xx. */
export class TransientFilesetError extends Error {}

const isTransient = (error: unknown): boolean => {
  if (!(error instanceof AxiosError)) return false;
  // A cancelled request (unmount, cancelQueries) must never be retried.
  if (error.code === AxiosError.ERR_CANCELED) return false;
  const status = error.response?.status;
  // No response at all means the request never completed — a network error or timeout.
  if (status === undefined) return true;
  return status === 429 || status >= 500;
};

/** Rethrows retryable failures as {@link TransientFilesetError} so the retry policy can see them. */
const withTransientTagging = async <T>(operation: () => Promise<T>, what: string): Promise<T> => {
  try {
    return await operation();
  } catch (error) {
    if (isTransient(error)) {
      const reason = error instanceof AxiosError ? error.message : 'request failed';
      throw new TransientFilesetError(`Failed to ${what}: ${reason}`);
    }
    throw error;
  }
};

/**
 * Selects the files holding a dataset split, in split order.
 *
 * HuggingFace shard names (`train-00000-of-00003-<hash>.parquet`) sort lexicographically into
 * split order, so a plain path sort is the shard order. Every rejection names what it saw:
 * a pattern that matches nothing is the most likely real failure, and it has to be
 * self-diagnosing rather than surfacing later as a confusing row shortfall.
 */
export const matchDatasetFiles = (
  files: FilesetFileOutput[],
  pattern: RegExp
): FilesetFileOutput[] => {
  // `g` and `y` both carry `lastIndex` between `test` calls, which would match every other
  // file rather than erroring — a wrong shard set is worse than a loud failure.
  const matcher =
    pattern.global || pattern.sticky
      ? new RegExp(pattern.source, pattern.flags.replace(/[gy]/g, ''))
      : pattern;

  const matches = files
    .filter((file) => matcher.test(file.path))
    .sort((a, b) => a.path.localeCompare(b.path));

  if (matches.length === 0) {
    const present = files.map((file) => file.path).join(', ');
    throw new Error(
      `No dataset file matched ${String(pattern)}. The repository holds: ${present || 'no files'}.`
    );
  }

  const notParquet = matches.find((file) => !file.path.toLowerCase().endsWith('.parquet'));
  if (notParquet) {
    throw new Error(
      `Expected a Parquet dataset file, but ${String(pattern)} matched ${notParquet.path}.`
    );
  }

  return matches;
};

/** Decodes the first `rowEnd` rows of a Parquet blob into plain objects. */
export const readParquetRows = async (
  blob: Blob,
  rowEnd: number
): Promise<Record<string, unknown>[]> => {
  try {
    const file = await blob.arrayBuffer();
    return await parquetReadObjects({ file, rowEnd });
  } catch (error) {
    logger.error('Failed to decode Parquet dataset file', error);
    throw new Error(
      'Could not read the dataset file. It is not valid Parquet, or uses a compression codec the browser cannot decode.'
    );
  }
};

interface FilesetRowsParams {
  workspace: string;
  filesetName: string;
  /** Matched against fileset-relative paths to find the split's shards. */
  pattern: RegExp;
  /** How many rows the caller needs. Shards are read in order until this is met. */
  rowCount: number;
  /** Bytes downloaded so far, against the total across the shards read so far. */
  onDownloadProgress?: (loadedBytes: number, totalBytes: number) => void;
}

const downloadShard = async (
  workspace: string,
  filesetName: string,
  path: string,
  onProgress: ((loadedBytes: number) => void) | undefined,
  signal: AbortSignal
): Promise<Blob> => {
  // `filesDownloadFile` would do, but it gives no progress. `customFetch` extends
  // AxiosRequestConfig, so onDownloadProgress passes straight through to axios.
  const [url] = getFilesDownloadFileQueryKey(
    encodeURIComponent(workspace),
    encodeURIComponent(filesetName),
    encodeURIComponent(path)
  );
  return customFetch<Blob>({
    url,
    method: 'GET',
    responseType: 'blob',
    signal,
    ...(onProgress ? { onDownloadProgress: (event) => onProgress(event.loaded) } : {}),
  });
};

/**
 * Reads dataset rows out of a fileset through the files service.
 *
 * The service proxies external (HuggingFace) storage server-side, so this works from a
 * browser that cannot reach huggingface.co itself — which is the whole point of going
 * through a fileset rather than calling the HuggingFace datasets server directly.
 */
export const fetchFilesetRows = async (
  { workspace, filesetName, pattern, rowCount, onDownloadProgress }: FilesetRowsParams,
  signal: AbortSignal
): Promise<Record<string, unknown>[]> => {
  const listing = await withTransientTagging(
    () => filesListFilesetFiles(workspace, filesetName, undefined, signal),
    'list the dataset files'
  );

  const shards = matchDatasetFiles(listing.data ?? [], pattern);

  const rows: Record<string, unknown>[] = [];
  let completedBytes = 0;
  let shardsRead = 0;

  for (const shard of shards) {
    if (rows.length >= rowCount) break;

    shardsRead += 1;
    if (shardsRead > MAX_DATASET_SHARDS) {
      throw new Error(
        `Reading ${rowCount} rows needed more than ${MAX_DATASET_SHARDS} shards of ${String(pattern)}.`
      );
    }

    // The denominator covers the shards proven necessary so far. It can only grow, and only
    // when a shard turns out not to have held enough rows — which no shipped recipe hits.
    const totalBytes = completedBytes + shard.size;
    if (totalBytes > MAX_DATASET_BYTES) {
      throw new Error(
        `Reading ${shard.path} would pull ${Math.ceil(totalBytes / 1024 / 1024)}MB, above the ${MAX_DATASET_BYTES / 1024 / 1024}MB limit.`
      );
    }

    const blob = await withTransientTagging(
      () =>
        downloadShard(
          workspace,
          filesetName,
          shard.path,
          onDownloadProgress
            ? (loaded) =>
                onDownloadProgress(Math.min(completedBytes + loaded, totalBytes), totalBytes)
            : undefined,
          signal
        ),
      `download ${shard.path}`
    );

    rows.push(...(await readParquetRows(blob, rowCount - rows.length)));
    completedBytes += shard.size;
    onDownloadProgress?.(completedBytes, completedBytes);
  }

  return rows;
};

/**
 * Cached rows for one fileset/pattern/count. `staleTime: Infinity` because a fileset is
 * pinned to an immutable revision, but `gcTime` is bounded — these rows are megabytes, and
 * holding them for a whole session after the template flow ends is pure waste.
 */
export const filesetRowsQueryOptions = (params: FilesetRowsParams) =>
  queryOptions({
    queryKey: [
      'fileset-parquet-rows',
      params.workspace,
      params.filesetName,
      String(params.pattern),
      params.rowCount,
    ],
    queryFn: ({ signal }) => fetchFilesetRows(params, signal),
    staleTime: Infinity,
    gcTime: 5 * 60 * 1000,
    retry: (failureCount: number, error: Error) =>
      failureCount < MAX_RETRIES && error instanceof TransientFilesetError,
    retryDelay: (attemptIndex: number) => RETRY_BASE_DELAY_MS * 2 ** attemptIndex,
  });
