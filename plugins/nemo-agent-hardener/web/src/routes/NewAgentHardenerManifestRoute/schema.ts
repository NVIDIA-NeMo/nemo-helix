// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';

/** Manifest ids are entity names: lowercase letters, digits and hyphens, not leading with a hyphen. */
export const NAME_PATTERN = /^[a-z0-9][a-z0-9-]*$/;

/**
 * The deployed-agent form. Every field but `name` is optional and free text: the list-ish ones
 * (`egress`, `secrets`, `env`) are parsed on submit, and `port` is validated as a number there so a
 * partially-typed value does not error while the user is still typing.
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
});

export type ManifestFormData = z.infer<typeof manifestFormSchema>;
