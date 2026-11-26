// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';

export const PURPOSE_LABELS: Record<FilesetPurpose, string> = {
  [FilesetPurpose.generic]: 'Generic',
  [FilesetPurpose.dataset]: 'Dataset',
  [FilesetPurpose.model]: 'Model',
  [FilesetPurpose.environment]: 'Environment',
};
