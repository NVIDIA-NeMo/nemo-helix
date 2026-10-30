// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';

/** Per-purpose copy shown in the purpose selector. Kept adjacent to the enum so each value has user-facing explanation. */
export const PURPOSE_OPTIONS: {
  value: FilesetPurpose;
  label: string;
  description: string;
}[] = [
  {
    value: FilesetPurpose.generic,
    label: 'Generic',
    description:
      "Default. Use for files that don't fit the Dataset or Model categories. Doesn't add purpose-specific metadata fields.",
  },
  {
    value: FilesetPurpose.dataset,
    label: 'Dataset',
    description:
      'For training and evaluation data. Enables dataset-specific metadata, including schema information.',
  },
  {
    value: FilesetPurpose.model,
    label: 'Model',
    description:
      'For model weights and checkpoints. Enables model-specific metadata, including tool-calling and model configuration fields.',
  },
  {
    value: FilesetPurpose.environment,
    label: 'Environment',
    description: 'For executable environment packages consumed by sandboxed workloads.',
  },
];
