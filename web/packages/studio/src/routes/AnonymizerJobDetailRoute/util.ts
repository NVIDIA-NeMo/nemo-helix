// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RunJob } from '@nemo/sdk/generated/anonymizer/schema';
import type { DataFileRow } from '@studio/components/FileRowEditor/types';

export const ANONYMIZER_POLLING_INTERVAL_MS = 5000;

export const jobStrategy = (job: RunJob): string | undefined => {
  const config = job.spec?.request?.config;
  if (!config) return undefined;
  if (config.rewrite) return 'rewrite';
  return (config.replace as { kind?: string } | undefined)?.kind;
};

export const jobSource = (job: RunJob): string | undefined => job.spec?.request?.data?.source;

export const RESULT_PREVIEW_ROWS = 20;

export interface ArtifactLocation {
  readonly fileset: string;
  readonly basePath: string;
}

/** `default/job-fileset-x#results/attempt-1/artifacts` → fileset `job-fileset-x`, path `results/…`. */
export const parseArtifactUrl = (artifactUrl: string | undefined): ArtifactLocation | undefined => {
  const [reference, basePath] = artifactUrl?.split('#') ?? [];
  if (!reference || !basePath) return undefined;
  const fileset = reference.split('/').pop();
  return fileset ? { fileset, basePath } : undefined;
};

/** Falls back to the row's first field when `metadata.json` didn't name a text column. */
export const resolveTextColumn = (row: DataFileRow, textColumn: string | undefined): string =>
  textColumn ?? Object.keys(row)[0] ?? '';

export const metadataTextColumn = (metadata: string | undefined): string | undefined => {
  try {
    const parsed: unknown = JSON.parse(metadata ?? '{}');
    const column = (parsed as { original_text_column?: unknown })?.original_text_column;
    return typeof column === 'string' ? column : undefined;
  } catch {
    return undefined;
  }
};
