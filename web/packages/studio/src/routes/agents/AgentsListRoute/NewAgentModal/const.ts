// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeploymentModeAvailabilityMode } from '@studio/api/agents/useDeploymentModes';
import { z } from 'zod';

export const AGENT_CONFIG_FILENAME = 'agent.yaml';

export const FABRIC_CONFIG_FORMAT = 'nemo-agents-spec-v1';

// Container staging skips this file, so its bytes never reach a deployment.
export const AGENT_SPEC_FILENAME = 'AGENT-SPEC.md';

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
  .trim()
  .min(1, 'Name is required')
  .regex(/^[a-z0-9]([a-z0-9-]*[a-z0-9])?$/, 'Use lowercase letters, numbers, and hyphens');

export const uploadAgentFormSchema = z.object({
  name: agentNameSchema,
  repoUrl: z.string().trim().default(''),
  secretKey: z.string().default(''),
  sshKeySecret: z.string().default(''),
  deploy: z.boolean().default(true),
  deploymentMode: z.nativeEnum(DeploymentModeAvailabilityMode).default('subprocess'),
});

export const UPLOAD_AGENT_FORM_DEFAULTS = {
  name: '',
  ...uploadAgentFormSchema.omit({ name: true }).parse({}),
};

/** Shown in the Repository field's info popover. */
export const REPOSITORY_EXAMPLES: readonly { label: string; value: string }[] = [
  { label: 'GitHub repository', value: 'github.com/acme/agents' },
  { label: 'GitHub branch and directory', value: 'github.com/acme/agents@main#agents/support' },
  { label: 'SSH remote', value: 'git@gitlab.example.com:acme/agents.git' },
  {
    label: 'SSH on a custom port, with a branch',
    value: 'ssh://git@gitlab.example.com:2222/acme/agents.git@release/1.0',
  },
  {
    label: 'SSH with a tag and directory',
    value: 'git@gitlab.example.com:acme/agents.git@v2.1#agents/support',
  },
];
