// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DATASET_NAME_REGEX } from '@studio/constants/constants';
import { z } from 'zod';

export const projectNameInputSchema = z
  .string()
  .min(1, { message: 'Workspace name is required' })
  .max(255)
  .regex(
    /^[\w.-]*$/,
    'Name must only contain alphanumeric characters, dashes, underscores, or dots.'
  )
  .regex(/[a-zA-Z0-9]/, 'Name must contain at least one alphanumeric character.');

export const URN_REGEX = /^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$/;
export const URNSchema = z
  .string()
  .min(1, { message: 'URN is required' })
  .regex(URN_REGEX, 'URN must be in the format of <namespace>/<name>');

export const datasetSchema = z.object({
  name: z
    .string()
    .min(1, { message: 'Name is required' })
    .regex(
      DATASET_NAME_REGEX,
      'Name must only contain alphanumeric characters, dashes, underscores, or dots'
    ),
  /** @deprecated Namespace is now provided via workspace at the API level */
  namespace: z
    .string()
    .regex(
      DATASET_NAME_REGEX,
      'Namespace must only contain alphanumeric characters, dashes, underscores, or dots'
    )
    .optional(),
  description: z.string(),
});
