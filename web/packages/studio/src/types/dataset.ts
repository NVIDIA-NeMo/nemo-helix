// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FileFormatType } from '@nemo/common/src/types';
import type {
  FileFormatDetectionResult,
  FileValidationResult,
} from '@nemo/common/src/utils/fileValidation';

export interface DatasetKeyMapping {
  promptKey: string | null;
  completionKey: string | null;
  idealResponseKey: string | null;
}

export interface DatasetInputFileResult {
  /** The dataset file URL (workspace/name#path format) */
  fileUrl: string;
  /** File format (json or jsonl) */
  format: FileFormatType;
  /** Validation result from file format check */
  validationResult: FileValidationResult;
  /** Schema detection result (chat-completion, completion, or unknown) */
  detectionResult: FileFormatDetectionResult;
  /** Resolved key mapping for prompt/completion/idealResponse */
  keyMapping: DatasetKeyMapping;
  /** Available keys extracted from the first row (for dropdowns) */
  availableKeys: Array<{ label: string; value: string }>;
  /** The parsed first row of data */
  firstRow: Record<string, unknown>;
  /** All parsed rows from the file */
  parsedRows: Record<string, unknown>[];
  /** Total number of rows in the file */
  rowCount: number;
}
