// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileFormat, InputFileSchemaType } from '@nemo/common/src/types';
import { extractUserFriendlyKeysFromRow, resolveKeyPath } from '@nemo/common/src/utils/file';
import { detectFileStructure, validateFileFormat } from '@nemo/common/src/utils/fileValidation';
import { type FileSampleMethod, sampleIndices } from '@nemo/common/src/utils/sampleTextLines';
import type { PromptRow } from '@studio/components/ModelComparePrompts/types';
import type { DatasetInputFileResult } from '@studio/types/dataset';

/** Builds prompt rows from parsed dataset rows using the shared sampling controls. */
export function buildPromptRowsFromParsedRows(
  fileResult: DatasetInputFileResult,
  sampleSize: number,
  sampleMethod: FileSampleMethod
): PromptRow[] {
  const promptKey = fileResult.keyMapping.promptKey;
  if (!promptKey || !fileResult.parsedRows?.length) return [];

  const parsedRows = fileResult.parsedRows;
  const indices = sampleIndices(parsedRows.length, sampleMethod, Math.max(1, sampleSize));

  const rows: PromptRow[] = [];
  for (const idx of indices) {
    const row = parsedRows[idx];
    if (!row) continue;
    const promptValue = resolveKeyPath(row, promptKey);
    if (promptValue === null || promptValue === undefined) continue;
    const prompt = typeof promptValue === 'string' ? promptValue : JSON.stringify(promptValue);
    rows.push({
      sourceIndex: idx,
      prompt,
      responses: {},
    });
  }
  return rows;
}

/**
 * Inline upload parser. Runs without a full validation UI — errors surface as a
 * small inline banner under the picker, because a single dropdown owns both
 * sample selection and upload.
 */
export async function parseUploadedFile(
  file: File
): Promise<DatasetInputFileResult | { error: string }> {
  const validation = await validateFileFormat(file);
  if (!validation.isValid || !validation.format) {
    return { error: validation.error ?? 'Invalid file format' };
  }
  const detection = await detectFileStructure(file, validation.format);
  const text = await file.text();
  let parsedRows: Record<string, unknown>[];
  try {
    if (validation.format === FileFormat.JSONL) {
      parsedRows = text
        .trim()
        .split('\n')
        .filter((line) => line.length > 0)
        .map((line) => JSON.parse(line) as Record<string, unknown>);
    } else {
      const parsed: unknown = JSON.parse(text);
      parsedRows = Array.isArray(parsed)
        ? (parsed as Record<string, unknown>[])
        : [parsed as Record<string, unknown>];
    }
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Failed to parse file contents' };
  }
  if (parsedRows.length === 0) {
    return { error: 'File contains no rows' };
  }
  const firstRow = (detection?.firstRow as Record<string, unknown> | undefined) ?? parsedRows[0];
  const availableKeys = firstRow ? extractUserFriendlyKeysFromRow(firstRow) : [];

  // Auto-detect prompt key: prefer the detector's answer, then fall back to common keys.
  let promptKey: string | null = null;
  if (detection?.schemaType === InputFileSchemaType.COMPLETION) {
    promptKey = detection.detectedFields.prompt ?? null;
  } else if (detection?.schemaType === InputFileSchemaType.CHAT_COMPLETION) {
    promptKey = detection.detectedMessages.user?.selector ?? null;
  }
  if (!promptKey && firstRow && typeof firstRow === 'object') {
    const candidates = ['prompt', 'question', 'input', 'text'];
    promptKey = candidates.find((k) => typeof firstRow[k] === 'string') ?? null;
  }
  // If detection couldn't find a prompt column we still return the parsed file
  // (with `promptKey: null`) so the inline column picker can let the user choose.
  return {
    fileUrl: `upload://${file.name}`,
    format: validation.format,
    validationResult: validation,
    detectionResult: detection,
    availableKeys,
    keyMapping: { promptKey, completionKey: null, idealResponseKey: null },
    firstRow,
    parsedRows,
    rowCount: parsedRows.length,
  };
}
