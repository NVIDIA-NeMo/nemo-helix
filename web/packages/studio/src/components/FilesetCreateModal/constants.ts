// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { FilesCreateFilesetBody } from '@nemo/sdk/generated/platform/zod/files';
import { z } from 'zod';

export enum StorageMode {
  Local = 'local',
  External = 'external',
}

export type SupportedPurpose = typeof FilesetPurpose.dataset | typeof FilesetPurpose.model;

// Same pattern as the generated zod, but reports which rule the value breaks.
export const filesetCreateFormSchema = FilesCreateFilesetBody.pick({
  name: true,
  description: true,
}).extend({
  name: z
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
    .transform((value) => toValidEntityName(value, value)),
  url: z.string().optional(),
  secretKey: z.string().optional(),
});

export type FilesetCreateFormData = z.infer<typeof filesetCreateFormSchema>;

export const PURPOSE_COPY: Record<SupportedPurpose, { title: string; submit: string }> = {
  [FilesetPurpose.dataset]: {
    title: 'Create Dataset',
    submit: 'Create Dataset',
  },
  [FilesetPurpose.model]: {
    title: 'Create Model Fileset',
    submit: 'Create Model Fileset',
  },
};
