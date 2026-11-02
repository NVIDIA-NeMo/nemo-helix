// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { deployOnCreateFormSchema } from '@studio/api/agents/useDeployOnCreate';
import { z } from 'zod';

export const AGENT_CONFIG_FILENAME = 'agent.yaml';

export const FABRIC_CONFIG_FORMAT = 'nemo-agents-spec-v1';

// Container staging skips this file, so its bytes never reach a deployment.
export const AGENT_SPEC_FILENAME = 'AGENT-SPEC.md';

export const AGENT_ETHOS_FILE = 'ETHOS.md';

// Mirrors MAX_AGENT_SPEC_STAGED_BYTES / _FILES; the platform only enforces them at deploy.
export const MAX_AGENT_SPEC_BYTES = 900_000;
export const MAX_AGENT_SPEC_FILES = 500;

// A directory picker hands over every descendant, so a mistaken pick can arrive with
// hundreds of thousands of entries. Reject on the raw count before mapping, filtering or
// sorting any of them — the ignore list cannot be applied without touching every entry.
export const MAX_PICKED_FILES = 1_000;

export const IGNORED_DIRECTORIES = new Set([
  '__pycache__',
  '.git',
  '.venv',
  'venv',
  'node_modules',
  '.mypy_cache',
  '.pytest_cache',
  '.ruff_cache',
  '.idea',
  '.vscode',
]);

export const IGNORED_FILENAMES = new Set(['.DS_Store', 'Thumbs.db']);

export const IGNORED_EXTENSIONS = ['.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll'];

export const agentNameSchema = z
  .string()
  .superRefine((value, ctx) => {
    if (sanitizeEntityName(value) === undefined) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: value
          ? 'Name must start with a letter and be at least 2 characters.'
          : 'Name is required.',
      });
    }
  })
  .transform((value) => toValidEntityName(value, value));

export const uploadAgentFormSchema = z
  .object({
    name: agentNameSchema,
    repoUrl: z.string().trim().default(''),
    secretKey: z.string().default(''),
  })
  .merge(deployOnCreateFormSchema);

export const UPLOAD_AGENT_FORM_DEFAULTS = {
  name: '',
  ...uploadAgentFormSchema.omit({ name: true }).parse({}),
};
