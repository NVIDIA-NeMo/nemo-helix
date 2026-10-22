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

/** The Purpose multi-select value for the default selection: `{ dataset: true, ... }`. */
export const DEFAULT_PURPOSE_FILTER = Object.fromEntries(
  VISIBLE_PURPOSES.map((purpose) => [purpose, true])
);

/**
 * Purposes ticked in a Purpose multi-select value (`{ [purpose]: true }`). Only entries that are
 * exactly `true` count. A bare string is the legacy single-select value kept in saved URLs.
 */
export const checkedPurposes = (value: unknown): FilesetPurpose[] => {
  if (typeof value === 'string') return value ? [value as FilesetPurpose] : [];
  return Object.entries((value ?? {}) as Record<string, unknown>)
    .filter(([, checked]) => checked === true)
    .map(([purpose]) => purpose as FilesetPurpose);
};

export const isDefaultPurposeFilter = (value: unknown) => {
  const checked = checkedPurposes(value);
  return (
    checked.length === VISIBLE_PURPOSES.length &&
    VISIBLE_PURPOSES.every((purpose) => checked.includes(purpose))
  );
};

export const PURPOSE_TAG_COLORS: Record<FilesetPurpose, TagColor> = {
  [FilesetPurpose.dataset]: 'blue',
  [FilesetPurpose.environment]: 'green',
  [FilesetPurpose.model]: 'purple',
  [FilesetPurpose.generic]: 'gray',
};
