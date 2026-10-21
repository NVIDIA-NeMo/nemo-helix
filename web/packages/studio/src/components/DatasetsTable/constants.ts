// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { type Tag } from '@nvidia/foundations-react-core';
import { type ComponentProps } from 'react';

export const PURPOSE_LABELS: Record<FilesetPurpose, string> = {
  [FilesetPurpose.generic]: 'Generic',
  [FilesetPurpose.dataset]: 'Dataset',
  [FilesetPurpose.model]: 'Model',
  [FilesetPurpose.environment]: 'Environment',
};

type TagColor = NonNullable<ComponentProps<typeof Tag>['color']>;

/**
 * Purposes shown by default. "generic" filesets are machine-generated (e.g. job logs), so the
 * Purpose filter leaves them off until asked for.
 */
export const VISIBLE_PURPOSES = [
  FilesetPurpose.dataset,
  FilesetPurpose.environment,
  FilesetPurpose.model,
] as const;

export const PURPOSE_TAG_COLORS: Record<FilesetPurpose, TagColor> = {
  [FilesetPurpose.dataset]: 'blue',
  [FilesetPurpose.environment]: 'green',
  [FilesetPurpose.model]: 'purple',
  [FilesetPurpose.generic]: 'gray',
};
