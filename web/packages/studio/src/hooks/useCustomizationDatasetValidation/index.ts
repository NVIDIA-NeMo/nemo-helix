// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { validateFileFormat } from '@nemo/common/src/utils/fileValidation';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { datasetFileContentQueryOptions } from '@studio/api/datasets/useDatasetFileContent';
import {
  datasetFileEncodingQueryOptions,
  type FileEncodingResult,
} from '@studio/hooks/useCustomizationDatasetValidation/encodingQuery';
import { parseFilesetUri } from '@studio/hooks/useCustomizationFiles/utils';
import { useDatasetFileDiscovery } from '@studio/hooks/useDatasetFileDiscovery';
import {
  type CustomizerSchemaDetection,
  type TrainingType,
  detectCustomizerSchema,
  expectedSchemaCopy,
  inferRowSchema,
  validateRowCompleteness,
} from '@studio/util/customizerSchema';
import { useQuery, useQueryClient } from '@tanstack/react-query';

export interface FileValidationError {
  path: string;
  error: string;
}

export interface CompletenessError {
  /** Fileset-relative path of the offending file. */
  path: string;
  /** 1-based row index for human display. */
  row: number;
  /** Short description of the missing/empty field. */
  message: string;
}

export interface EncodingFileError {
  path: string;
}

/**
 * SDK file output decorated with the per-file row count this hook computes
 * during validation. rowCount is undefined while validation is in flight or
 * when validation skipped the file (e.g., format check failed).
 */
export interface AnnotatedFilesetFile extends FilesetFileOutput {
  rowCount?: number;
}

/** Per-file completeness check results are capped to keep the UI fast. */
const MAX_COMPLETENESS_ERRORS_PER_FILE = 10;

/**
 * Cap concurrent per-file fetch+validate operations. Each file kicks off two
 * requests (content + strict UTF-8 encoding check) plus buffers a full
 * arrayBuffer transiently — without a cap, a 100-file fileset would issue
 * 200 concurrent requests and could buffer hundreds of MB. Browser HTTP/1.1
 * pools cap at ~6 per origin anyway, so high parallelism just queues without
 * benefit. 4 leaves headroom for other in-flight queries on the page.
 */
const PER_FILE_VALIDATION_CONCURRENCY = 4;

/**
 * Run an async worker over an items array with bounded concurrency, preserving
 * input order in the result. Equivalent to Promise.all(items.map(worker)) but
 * with at most `concurrency` workers active at any time.
 */
const runWithConcurrency = async <T, R>(
  items: T[],
  worker: (item: T) => Promise<R>,
  concurrency: number
): Promise<R[]> => {
  const results: R[] = new Array(items.length);
  let cursor = 0;
  const runOne = async (): Promise<void> => {
    while (true) {
      const i = cursor++;
      if (i >= items.length) return;
      results[i] = await worker(items[i]);
    }
  };
  const lanes = Array.from({ length: Math.min(concurrency, items.length) }, runOne);
  await Promise.all(lanes);
  return results;
};

export interface CustomizationDatasetValidationResult {
  isPending: boolean;
  /**
   * Non-null when the fileset's file listing itself failed (network/API/perms).
   * Callers should render a retryable load error instead of any of the other
   * checks — the rest of the result is meaningless when discovery failed.
   */
  discoveryError: Error | null;
  /** Format check across every discovered file. */
  format: {
    ok: boolean;
    fileErrors: FileValidationError[];
  };
  /**
   * Strict customizer-aligned schema detection across discovered files. Set
   * from the FIRST file that matched; null when no file matched.
   */
  schema: CustomizerSchemaDetection | null;
  /**
   * Human-readable description of the keys customizer expects for the active
   * training type, surfaced in the "Schema does not match" copy.
   */
  schemaExpectedCopy: string;
  /**
   * Files that parsed cleanly but whose first row did not match any
   * customizer-recognized shape, OR matched a different variant than
   * `schema.variant`. Customizer would reject these at training time, so the
   * panel must surface them rather than silently relying on the first match.
   */
  schemaMismatchedFiles: string[];
  /**
   * Completeness check (no empty/null values in required fields). Skipped when
   * format failed or the schema didn't match — we don't know what to require
   * in those cases.
   */
  completeness: {
    /** True when the check ran and found no errors. */
    ok: boolean;
    /** True when the check was skipped (format failed or schema didn't match). */
    skipped: boolean;
    /** Per-file errors, capped at MAX_COMPLETENESS_ERRORS_PER_FILE per file. */
    errors: CompletenessError[];
  };
  /**
   * Strict UTF-8 encoding check across every discovered file. Uses
   * `TextDecoder('utf-8', { fatal: true })` on raw bytes, which throws on the
   * same input customizer's Python `open(encoding="utf-8")` would reject.
   */
  encoding: {
    ok: boolean;
    fileErrors: EncodingFileError[];
  };
  /** TS-shaped string of the first recognized row's inferred schema; empty when none. */
  schemaShape: string;
  hasTraining: boolean;
  hasValidation: boolean;
  /**
   * Root-level .json files that matched no training/validation pattern.
   * Not validated by the main pass (they are not in `training`).
   * Only native Unsloth with apply_chat_template disabled should treat these
   * as training; all other consumers can ignore this field.
   */
  unmatchedRootJson: FilesetFileOutput[];
  /** Show "customizer will auto-split 10%" notice. */
  autoSplitNotice: boolean;
  /**
   * Discovered training files, each annotated with its non-empty row count.
   * The count is undefined while validation is pending or the file was
   * skipped — consumers should fall back to "no count yet" rendering.
   */
  training: AnnotatedFilesetFile[];
  /** Discovered validation files, decorated the same way as training. */
  validation: AnnotatedFilesetFile[];
  /** Total non-empty rows across all training files (for auto-split stats). */
  trainingRowCount: number;
  /** Total non-empty rows across all validation files. */
  validationRowCount: number;
}

interface UseCustomizationDatasetValidationOptions {
  fileset?: string;
  /**
   * Currently-selected training type from the form. Schema rules differ:
   * SFT accepts messages or prompt+completion; DPO accepts the four preference
   * shapes. Customizer applies the same training-type-aware discrimination.
   */
  trainingType: TrainingType;
  /**
   * Cap on the number of lines parsed per file. 0 (default) means parse every
   * line of every file. Positive values truncate the sample, useful for
   * lightweight pre-flight callers.
   */
  sampleLimit?: number;
}

interface PerFileValidation {
  file: FilesetFileOutput;
  error: string | null;
  schema: CustomizerSchemaDetection | null;
  firstRow: Record<string, unknown> | null;
  /** Non-empty line count from the full content (independent of sampleLimit). */
  rowCount: number;
  /**
   * Completeness errors collected for this file, capped at
   * MAX_COMPLETENESS_ERRORS_PER_FILE. Empty when the schema didn't detect
   * (we can't check) or when every row is complete.
   */
  completenessErrors: CompletenessError[];
  /** Strict UTF-8 check result for this file's raw bytes. */
  encoding: FileEncodingResult;
}

const buildSampleFile = (path: string, content: string, sampleLimit: number): File => {
  const text = sampleLimit > 0 ? content.split('\n').slice(0, sampleLimit).join('\n') : content;
  return new File([text], path, { type: 'application/x-ndjson' });
};

const countRows = (content: string): number =>
  content.split('\n').filter((line) => line.trim().length > 0).length;

/**
 * Iterate every non-empty line of JSONL content and call the visitor with the
 * parsed row. Returns true if all lines parsed; false on a parse error (caller
 * already runs validateFileFormat which would catch this, so this should only
 * happen if line counts diverge).
 */
const forEachParsedRow = (
  content: string,
  visitor: (row: Record<string, unknown>, index: number) => 'stop' | void
): void => {
  let index = 0;
  for (const line of content.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        const result = visitor(parsed as Record<string, unknown>, index);
        if (result === 'stop') return;
      }
    } catch {
      return;
    }
    index += 1;
  }
};

const validateOne = async (
  file: FilesetFileOutput,
  content: string,
  encoding: FileEncodingResult,
  sampleLimit: number,
  trainingType: TrainingType
): Promise<PerFileValidation> => {
  const rowCount = countRows(content);
  const sample = buildSampleFile(file.path, content, sampleLimit);
  const formatResult = await validateFileFormat(sample);
  if (!formatResult.isValid || !formatResult.format) {
    return {
      file,
      error: formatResult.error ?? 'File is not valid JSON or JSONL',
      schema: null,
      firstRow: null,
      rowCount,
      completenessErrors: [],
      encoding,
    };
  }

  let firstRow: Record<string, unknown> | null = null;
  let schema: CustomizerSchemaDetection | null = null;
  const completenessErrors: CompletenessError[] = [];

  // Honor sampleLimit for the completeness scan as well, so lightweight
  // callers don't quietly walk every line of every file. Schema detection
  // only reads the first row regardless, so truncation is safe for it too.
  // rowCount above stays computed off the full content — that field exposes
  // true file size for the auto-split stats, independent of sample mode.
  const sampledContent =
    sampleLimit > 0 ? content.split('\n').slice(0, sampleLimit).join('\n') : content;

  forEachParsedRow(sampledContent, (row, index) => {
    if (firstRow === null) {
      firstRow = row;
      schema = detectCustomizerSchema(row, trainingType);
    }
    // Skip completeness when the schema didn't match — we don't know what to
    // require. The Schema check already surfaces a warning.
    if (schema === null) return 'stop';

    const message = validateRowCompleteness(row, schema.variant);
    if (message !== null) {
      completenessErrors.push({ path: file.path, row: index + 1, message });
      if (completenessErrors.length >= MAX_COMPLETENESS_ERRORS_PER_FILE) {
        return 'stop';
      }
    }
  });

  return {
    file,
    error: null,
    schema,
    firstRow,
    rowCount,
    completenessErrors,
    encoding,
  };
};

/**
 * Downloads every training/validation file in a fileset and runs format, schema
 * (scoped to the training type), encoding, and completeness checks.
 * sampleLimit defaults to 0 (parse every line); positive values truncate.
 */
export const useCustomizationDatasetValidation = ({
  fileset,
  trainingType,
  sampleLimit = 0,
}: UseCustomizationDatasetValidationOptions): CustomizationDatasetValidationResult => {
  const { workspace, name } = parseFilesetUri(fileset ?? '');
  const queryClient = useQueryClient();
  const {
    training,
    validation,
    unmatchedRootJson,
    isPending: isDiscoveryPending,
    error: discoveryError,
  } = useDatasetFileDiscovery({ fileset });

  const allFiles = [...training, ...validation];
  const paths = allFiles.map((f) => f.path);
  const enabled = !!workspace && !!name && allFiles.length > 0;

  const { data, isPending: isValidationPending } = useQuery({
    enabled,
    // Sort paths in the key so a backend reorder of the same files doesn't
    // change the key and spuriously refetch the whole validation pass.
    // allFiles below stays in the original order so per-file iteration
    // behavior is unchanged.
    queryKey: [
      'customization-dataset-validation',
      workspace,
      name,
      trainingType,
      sampleLimit,
      [...paths].sort(),
    ] as const,
    queryFn: async () => {
      const perFile = await runWithConcurrency<FilesetFileOutput, PerFileValidation>(
        allFiles,
        async (file): Promise<PerFileValidation> => {
          try {
            // Fetch lossy text + strict UTF-8 check in parallel for THIS
            // file (two requests per file). The outer concurrency cap
            // bounds how many files run at once. Folding the strict decode
            // into datasetFileContentQueryOptions is tracked as a follow-up.
            const [content, encoding] = await Promise.all([
              queryClient.ensureQueryData(
                datasetFileContentQueryOptions({ workspace, name, path: file.path })
              ),
              queryClient.ensureQueryData(
                datasetFileEncodingQueryOptions({ workspace, name, path: file.path })
              ),
            ]);
            return await validateOne(file, content, encoding, sampleLimit, trainingType);
          } catch (err) {
            // A single file failing to download (404, network blip, fileset
            // shifting underneath us) shouldn't poison the whole validation —
            // surface it on this file's row and let the rest be validated.
            const message = err instanceof Error ? err.message : 'Failed to download file';
            return {
              file,
              error: `Failed to download file: ${message}`,
              schema: null,
              firstRow: null,
              rowCount: 0,
              completenessErrors: [],
              // Don't double-count this in the encoding row — the format/error
              // row already carries the download failure for this file.
              encoding: { ok: true },
            };
          }
        },
        PER_FILE_VALIDATION_CONCURRENCY
      );
      return perFile;
    },
  });

  const perFile = data ?? [];
  const fileErrors: FileValidationError[] = perFile
    .filter((r) => r.error !== null)
    .map((r) => ({ path: r.file.path, error: r.error as string }));
  const formatOk = enabled && fileErrors.length === 0 && perFile.length === allFiles.length;
  const firstDetected = perFile.find((r) => r.schema !== null) ?? null;

  const hasTraining = training.length > 0;
  const hasValidation = validation.length > 0;

  // Decorate the discovered files with their per-file row counts so consumers
  // get one annotated array instead of having to merge a parallel path->count
  // map. rowCount is undefined for any file whose validation hasn't resolved
  // yet (or was skipped because format failed).
  const rowCountsByPath: Record<string, number> = Object.fromEntries(
    perFile.map((r) => [r.file.path, r.rowCount])
  );
  const annotate = (file: FilesetFileOutput): AnnotatedFilesetFile => ({
    ...file,
    rowCount: rowCountsByPath[file.path],
  });
  const annotatedTraining = training.map(annotate);
  const annotatedValidation = validation.map(annotate);

  const trainingPaths = new Set(training.map((f) => f.path));
  const trainingRowCount = perFile
    .filter((r) => trainingPaths.has(r.file.path))
    .reduce((sum, r) => sum + r.rowCount, 0);
  const validationRowCount = perFile
    .filter((r) => !trainingPaths.has(r.file.path))
    .reduce((sum, r) => sum + r.rowCount, 0);

  const schema = firstDetected?.schema ?? null;
  // A file mismatches when its format check passed but its first row either
  // didn't match any customizer shape OR matched a different variant than the
  // one we're treating as "the" schema. Customizer's per-file Pydantic
  // validation would reject these at training time.
  const schemaMismatchedFiles = perFile
    .filter(
      (r) =>
        r.error === null &&
        (r.schema === null || (schema !== null && r.schema?.variant !== schema.variant))
    )
    .map((r) => r.file.path);
  const completenessErrors = perFile.flatMap((r) => r.completenessErrors);
  const completenessSkipped = !formatOk || schema === null;
  const completeness = {
    ok: !completenessSkipped && completenessErrors.length === 0,
    skipped: completenessSkipped,
    errors: completenessErrors,
  };
  const encodingFileErrors: EncodingFileError[] = perFile
    .filter((r) => !r.encoding.ok)
    .map((r) => ({ path: r.file.path }));
  const encoding = {
    ok: enabled && encodingFileErrors.length === 0 && perFile.length === allFiles.length,
    fileErrors: encodingFileErrors,
  };

  return {
    isPending: isDiscoveryPending || (enabled && isValidationPending),
    discoveryError,
    format: { ok: formatOk, fileErrors },
    schema,
    schemaExpectedCopy: expectedSchemaCopy(trainingType),
    schemaMismatchedFiles,
    schemaShape: inferRowSchema(firstDetected?.firstRow ?? null),
    completeness,
    encoding,
    hasTraining,
    hasValidation,
    unmatchedRootJson,
    autoSplitNotice: hasTraining && !hasValidation,
    training: annotatedTraining,
    validation: annotatedValidation,
    trainingRowCount,
    validationRowCount,
  };
};
