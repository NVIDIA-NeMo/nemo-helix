// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { entityNameSchema } from '@nemo/common/src/utils/entityName';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { FilesCreateFilesetBody } from '@nemo/sdk/generated/platform/zod/files';
import { z } from 'zod';

// Same pattern as the generated zod, but reports which rule the value breaks.
export const DatasetCreateFilesetFormSchema = FilesCreateFilesetBody.extend({
  name: z.string().trim().pipe(entityNameSchema('Name')),
  purpose: z.nativeEnum(FilesetPurpose),
});

export type CreateFilesetFormFields = z.infer<typeof DatasetCreateFilesetFormSchema>;

/** Form extends schema with optional files (Upload) and external storage inputs (url/secretKey). */
export type DatasetFormFields = CreateFilesetFormFields & {
  files?: File[];
  url?: string;
  secretKey?: string;
};
