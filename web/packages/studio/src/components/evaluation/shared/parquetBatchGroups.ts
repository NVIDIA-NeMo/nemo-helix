// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const PARQUET_EXTENSION = '.parquet';
const GLOB_CHARS = /[*?[\]]/;
/** Job filesets keep their log shards here, beside the job's actual output. */
const JOB_LOGS_PREFIX = 'logs/';

export interface ParquetBatches {
  /** Fileset-relative glob matching every Parquet file in the directory, as the evaluator resolves it. */
  glob: string;
  /** Directory holding the batches, with a trailing slash; empty at the fileset root. */
  dir: string;
  count: number;
  /** First batch in path order, read to validate the set. */
  firstPath: string;
}

/** Every directory holding more than one Parquet file, as a glob over those files. A directory
 *  whose name holds glob characters is skipped, since the evaluator would read them as wildcards. */
export const parquetBatchGroups = (filesetPaths: readonly string[]): ParquetBatches[] => {
  const byDir = new Map<string, string[]>();
  for (const path of filesetPaths) {
    if (!path.endsWith(PARQUET_EXTENSION) || path.startsWith(JOB_LOGS_PREFIX)) continue;
    const dir = path.slice(0, path.lastIndexOf('/') + 1);
    if (GLOB_CHARS.test(dir)) continue;
    byDir.set(dir, [...(byDir.get(dir) ?? []), path]);
  }

  return [...byDir]
    .filter(([, paths]) => paths.length > 1)
    .map(([dir, paths]) => ({
      glob: `${dir}*${PARQUET_EXTENSION}`,
      dir,
      count: paths.length,
      firstPath: [...paths].sort()[0],
    }))
    .sort((a, b) => a.dir.localeCompare(b.dir));
};
