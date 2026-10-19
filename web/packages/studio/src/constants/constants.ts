/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { Palette } from 'lucide-react';

export const CHAT_DEFAULT_MAX_TOKENS = 4096;
export const DEFAULT_LARGE_PAGE_SIZE = 1000;
export const DATASET_NAME_REGEX = /^[a-zA-Z0-9._-]+$/;
export const DEFAULT_MODEL_NAME = 'nvidia/nemotron-3.5-lightning-30b-a3b';
export const DEFAULT_NAMESPACE = 'default';
export const DEFAULT_API_ERR_MSG = 'Invalid API response. Please try again later.';
export const DEFAULT_TOOLS_FILE_NAME = 'tools.json';
export const EMPTY_FIELD_VALUE = '-';
export const EMPTY_FIELD_EMDASH_VALUE = '—';
/**
 * Model *entity* names, not served model names: the models service normalizes discovered model
 * ids to entity names (`normalize_model_entity_name`), turning `/` and `.` into `-` — so
 * `nvidia/nemotron-3.5-lightning-30b-a3b` is stored as `nvidia-nemotron-3-5-lightning-30b-a3b`.
 * Templates look models up by entity name, so a dotted name here never resolves.
 */
export const DEFAULT_BUILD_MODEL_NAME = 'nvidia-nemotron-3-5-lightning-30b-a3b';
export const DEFAULT_EMBEDDER_MODEL_NAME = 'nvidia-nv-embedqa-e5-v5';

/**
 * Engine for a guardrail config's `main` model entry.
 *
 * Mirrors `DEFAULT_MAIN_ENGINE` in
 * `plugins/nemo-guardrails/src/nemo_guardrails_plugin/constants.py`, which is what the
 * service falls back to when a config declares no `main` entry. Writing the same value
 * keeps a Studio-authored config behaviourally identical to one without the entry.
 */
export const GUARDRAIL_DEFAULT_ENGINE = 'nim';

export const DEFAULT_MAX_PARALLEL_REQUESTS = 2;
export const MAX_PARALLEL_REQUESTS_MIN = 1;
export const MAX_PARALLEL_REQUESTS_MAX = 64;

export const DEFAULT_TEXT_INFERENCE_PARAMS = {
  temperature: 0.7,
  top_p: 0.9,
  max_parallel_requests: DEFAULT_MAX_PARALLEL_REQUESTS,
} as const;

export const KNOWN_TEXT_EXTENSIONS = new Set([
  // Data
  'json',
  'jsonl',
  'csv',
  'tsv',
  // Code
  'py',
  'js',
  'jsx',
  'ts',
  'tsx',
  'java',
  'c',
  'cpp',
  'h',
  'go',
  'rs',
  'rb',
  'php',
  'swift',
  'kt',
  'scala',
  'r',
  'm',
  'sh',
  'bash',
  'zsh',
  'fish',
  // Markup / Config
  'html',
  'htm',
  'xml',
  'yaml',
  'yml',
  'toml',
  'ini',
  'cfg',
  'conf',
  'jsonc',
  'env',
  // Text
  'txt',
  'md',
  'rst',
  'log',
  'diff',
  'patch',
  // Other
  'sql',
  'graphql',
  'proto',
  'dockerfile',
  'makefile',
]);

export const DataDesignerIconFc = Palette;
