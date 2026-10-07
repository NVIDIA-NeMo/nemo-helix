// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const PARQUET_EXTENSION = '.parquet';
const GLOB_CHARS = /[*?[\]]/;

export interface ParquetBatches {
  /** Fileset-relative glob matching every Parquet file in the directory, as the evaluator resolves it. */
  glob: string;
  /** Directory holding the batches, with a trailing slash; empty at the fileset root. */
  dir: string;
  count: number;
}

/** The Parquet files sharing `path`'s directory, when there is more than one. A directory whose
 *  name holds glob characters is skipped, since the evaluator would read them as wildcards. */
export const parquetBatchGlob = (
  path: string,
  filesetPaths: readonly string[]
): ParquetBatches | null => {
  if (!path.endsWith(PARQUET_EXTENSION)) return null;
  const dir = path.slice(0, path.lastIndexOf('/') + 1);
  if (GLOB_CHARS.test(dir)) return null;

  const count = filesetPaths.filter(
    (candidate) =>
      candidate.startsWith(dir) &&
      !candidate.slice(dir.length).includes('/') &&
      candidate.endsWith(PARQUET_EXTENSION)
  ).length;
  return count > 1 ? { glob: `${dir}*${PARQUET_EXTENSION}`, dir, count } : null;
};
