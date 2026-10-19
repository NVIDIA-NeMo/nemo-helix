// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';

/** Manifest ids are entity names: lowercase letters, digits and hyphens, not leading with a hyphen. */
export const NAME_PATTERN = /^[a-z0-9][a-z0-9-]*$/;

/** Where the victim comes from: a registered agent, or an image whose author owns the Dockerfile. */
export type ManifestSource = 'agent' | 'project';

/**
 * The create form. Every field but `name` is optional and free text: the list-ish ones
 * (`egress`, `secrets`, `env`, `binaries`) are parsed on submit, and `port` is validated as a number
 * there so a partially-typed value does not error while the user is still typing.
 *
 * The project fields exist because a Dockerfile cannot state everything a manifest needs. They are
 * pre-filled from the upload wherever it can, so what remains on screen is only what the project
 * could not say about itself.
 */
export const manifestFormSchema = z.object({
  name: z
    .string()
    .trim()
    .min(1, 'A manifest id is required')
    .regex(NAME_PATTERN, 'Lowercase letters, digits and hyphens only'),
  agent: z.string().trim().optional(),
  egress: z.string().trim().optional(),
  env: z.string().trim().optional(),
  port: z.string().trim().optional(),
  secrets: z.string().trim().optional(),
  dockerfile: z.string().trim().optional(),
  startCommand: z.string().trim().optional(),
  binaries: z.string().trim().optional(),
  harness: z.string().trim().optional(),
});

export type ManifestFormData = z.infer<typeof manifestFormSchema>;
