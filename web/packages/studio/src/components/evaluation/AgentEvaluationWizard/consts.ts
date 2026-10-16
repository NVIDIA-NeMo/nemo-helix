// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const DATASET_FILE_ACCEPT = '.jsonl,.json,.csv,.parquet';

/** Job filesets keep their log shards here, beside the job's actual output. */
export const JOB_LOGS_PREFIX = 'logs/';

const EVAL_CONFIG_FILE_EXTENSIONS = ['.json', '.yaml', '.yml'];

export const EVAL_CONFIG_FILE_ACCEPT = EVAL_CONFIG_FILE_EXTENSIONS.join(',');

/** Whether a fileset path could hold an eval config. */
export const isEvalConfigCandidate = (path: string): boolean =>
  !path.startsWith(JOB_LOGS_PREFIX) &&
  EVAL_CONFIG_FILE_EXTENSIONS.some((extension) => path.toLowerCase().endsWith(extension));
