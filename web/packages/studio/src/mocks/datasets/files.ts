// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { CUSTOMIZATION_FILESET_FILEPATHS } from '@studio/constants/customization';

export const customizationFiles: FilesetFileOutput[] = [
  {
    size: 130,
    file_ref: 'ref-eval',
    path: 'eval_medical_input_array.json',
    file_url: 'https://example.com/eval_medical_input_array.json',
  },
  {
    size: 131,
    file_ref: 'ref-train',
    path: CUSTOMIZATION_FILESET_FILEPATHS.Training,
    file_url: 'https://example.com/training.json',
  },
  {
    size: 132,
    file_ref: 'ref-val',
    path: CUSTOMIZATION_FILESET_FILEPATHS.Validation,
    file_url: 'https://example.com/validation.json',
  },
];
