// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import {
  extractUserFriendlyKeysFromRow,
  findMessagesArray,
  getFileRowCount,
  getRowAtIndex,
} from '@nemo/common/src/utils/file';
import {
  useFilesListFilesetFiles,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import { useDownloadFileHead } from '@studio/components/filesets/hooks/useDownloadFileHead';
import { isSupportedMappingPath } from '@studio/routes/evaluation/EvaluationNewRoute/types';
import {
  detectFormatFromPath,
  resolveSchemaForFile,
} from '@studio/routes/FilesetDetailRoute/DatasetSchemaEditor/helpers';
import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';

/** Bytes read for line-delimited formats, where a partial tail still parses. */
const HEAD_BYTES = 65536;

export interface DatasetKeyOption {
  label: string;
  value: string;
}

/** One message in a detected messages array, as ``extractUserFriendlyKeysFromRow``
 *  reports it: ``label`` names the role, ``selector`` is the dot-bracket path
 *  (e.g. ``conversation[1].content``) carrying the real array index, so a
 *  leading system message shifts user to ``[1]`` and assistant to ``[2]``. */
export interface MessageSelector {
  label: string;
  selector: string;
  /** ``system`` | ``user`` | ``assistant``, taken from the message at that index. */
  role: string;
}

/** Real array index carried by a selector, e.g. ``messages[2].content`` -> 2. */
const selectorIndex = (selector: string): number =>
  Number(selector.match(/\[(\d+)\]/)?.[1] ?? Number.NaN);

/** The last assistant message and the user message directly before it, paired
 *  by selector index because contentless messages never reach ``selectors``. */
export const lastExchange = (
  selectors: MessageSelector[]
): { user: string | null; assistant: string | null } => {
  const assistant = selectors.filter((entry) => entry.role === 'assistant').at(-1) ?? null;
  if (!assistant) {
    return { user: lastSelectorForRole(selectors, 'user'), assistant: null };
  }
  const preceding = selectors.find(
    (entry) => selectorIndex(entry.selector) === selectorIndex(assistant.selector) - 1
  );
  return preceding?.role === 'user'
    ? { user: preceding.selector, assistant: assistant.selector }
    : { user: null, assistant: null };
};

/** The last message with a given role. */
const lastSelectorForRole = (selectors: MessageSelector[], role: string): string | null =>
  selectors.filter((entry) => entry.role === role).at(-1)?.selector ?? null;

export interface DatasetPreview {
  /** The row at ``rowIndex``. Row 0 is also the row the live test scores. */
  row: Record<string, unknown> | null;
  /** Rows reachable in the fetched head -- the range the pager can page over.
   *  Not the file's total unless ``isPartial`` is false. */
  rowCount: number;
  /** The file is larger than the fetched head, so ``rowCount`` undercounts it. */
  isPartial: boolean;
  /** Columns offerable as a ``field_mapping`` binding target. */
  keyOptions: DatasetKeyOption[];
  /** Column holding an OpenAI-style messages array, when row 0 has one. */
  messagesColumn: string | null;
  /** Role-labelled positional selectors into that array. */
  messageSelectors: MessageSelector[];
  isLoading: boolean;
  error: string | null;
}

/** Column names declared by a stored dataset schema's ``properties`` map. A
 *  column that is null in row 0 still shows up here, so these are merged in
 *  rather than relying on row 0 alone. */
const schemaPropertyKeys = (schema: unknown): string[] => {
  if (!schema || typeof schema !== 'object') return [];
  const properties = (schema as { properties?: unknown }).properties;
  if (!properties || typeof properties !== 'object' || Array.isArray(properties)) return [];
  return Object.keys(properties as Record<string, unknown>);
};

/**
 * Reads row 0 of the selected dataset file and derives the column paths the
 * mapping panel can bind to.
 *
 * Row 0 comes from a cached HTTP Range head — the same
 * ``useDownloadFileHead`` + ``getFirstRow`` path ``useDatasetSchemaEditor``
 * uses to infer schemas from existing files — so no bespoke sniffing is needed.
 * When the fileset carries a saved ``metadata.dataset`` schema, its declared
 * properties are merged in on top.
 */
export function useDatasetPreview(datasetRef: string | null, rowIndex = 0): DatasetPreview {
  const downloadFileHead = useDownloadFileHead();
  const parsed = datasetRef ? parseFilesetLocation(datasetRef) : null;
  const path = parsed?.objectPath ?? '';
  const format = path ? detectFormatFromPath(path) : null;

  const filesetEnabled = Boolean(parsed?.workspace && parsed?.name);

  const { data: fileset } = useFilesRetrieveFileset(parsed?.workspace ?? '', parsed?.name ?? '', {
    query: { enabled: filesetEnabled },
  });

  const { data: filesResponse } = useFilesListFilesetFiles(
    parsed?.workspace ?? '',
    parsed?.name ?? '',
    undefined,
    { query: { enabled: filesetEnabled } }
  );

  const fileSize = useMemo(
    () => filesResponse?.data?.find((file) => file.path === path)?.size,
    [filesResponse, path]
  );

  /** Byte budget for the Range request.
   *
   *  Never more than the file's real size: the Files endpoint rejects a range
   *  running past EOF, so a fixed 64 KB ask 416s on any smaller file -- which is
   *  most eval datasets. ``handleInferFromExisting`` passes the exact size for
   *  the same reason. Undefined until the listing resolves, which also gates the
   *  row query so it never fires with a bad budget.
   *
   *  A head is only safe for line-delimited formats. A truncated JSON array
   *  cannot be parsed at all, so ``.json`` is read whole -- the same thing
   *  ``handleInferFromExisting`` does for every format. */
  const fileBytes = useMemo(() => {
    if (fileSize === undefined) return undefined;
    return format === 'json' ? fileSize : Math.min(fileSize, HEAD_BYTES);
  }, [fileSize, format]);

  /** Whether the read covers the entire file, i.e. whether a row total is real. */
  const isComplete = useMemo(
    () => fileBytes !== undefined && fileBytes === fileSize,
    [fileBytes, fileSize]
  );

  const {
    data: preview,
    isFetching,
    error,
  } = useQuery({
    queryKey: ['evaluation-new', 'dataset-row', datasetRef, fileBytes, rowIndex],
    enabled: Boolean(parsed && path && format && fileBytes !== undefined),
    staleTime: Infinity,
    // A failed head is a dead end, not a flake: retrying just parks the user
    // behind a spinner for the whole backoff before showing the same error.
    retry: false,
    queryFn: async () => {
      if (!parsed || !format) return null;
      const buffer = await downloadFileHead({
        workspace: parsed.workspace,
        datasetName: parsed.name,
        path,
        bytes: fileBytes,
      });
      if (!buffer) {
        throw new Error('Could not read the selected file.');
      }
      const blob = new File([new TextDecoder('utf-8').decode(buffer)], path);
      return {
        row: await getRowAtIndex(blob, format, rowIndex),
        rowCount: await getFileRowCount(blob, format),
      };
    },
  });

  const row = preview?.row ?? null;

  /** Bindable targets are TOP-LEVEL column names only.
   *
   *  Deliberately not ``extractUserFriendlyKeysFromRow``: for a messages column
   *  that helper emits only ``messages[0].content``-style selectors and never
   *  the bare column, and ``FieldMapping`` refuses any path containing ``[`` or
   *  ``]`` ("array path segments are not supported for column mappings"). Using
   *  it here would leave an OpenAI-format dataset with nothing bindable at all.
   *
   *  The array is reached instead by binding the whole column to the canonical
   *  ``messages`` field and indexing it in the template, where brackets are
   *  legal — see ``messageSelectors``. */
  const keyOptions = useMemo(() => {
    const options: DatasetKeyOption[] = row
      ? Object.keys(row).map((key) => ({ label: key, value: key }))
      : [];
    const seen = new Set(options.map((option) => option.value));
    for (const key of schemaPropertyKeys(resolveSchemaForFile(fileset?.metadata?.dataset, path))) {
      if (!seen.has(key)) {
        seen.add(key);
        options.push({ label: key, value: key });
      }
    }
    return options;
  }, [row, fileset, path]);

  /** Role-labelled positional selectors for a detected messages array.
   *
   *  Sourced from ``extractUserFriendlyKeysFromRow`` so index derivation stays
   *  in one place: it walks the raw array, so a leading system message shifts
   *  user to ``[1]`` and assistant to ``[2]``.
   *
   *  Positional, not role-matched, on purpose: a filter chain like
   *  ``{{ (messages | selectattr('role','equalto','user') | list | last).content }}``
   *  renders correctly but fails ``input_schema()`` with
   *  ``TemplateSchemaInferenceError: unsupported Jinja expression for dataset
   *  schema inference: Filter``, so it cannot be used in a metric. */
  const { messagesColumn, messageSelectors } = useMemo(() => {
    const found = row ? findMessagesArray(row) : null;
    if (!row || !found) {
      return { messagesColumn: null, messageSelectors: [] as MessageSelector[] };
    }
    const selectors = extractUserFriendlyKeysFromRow(row, found)
      .filter((option) => !isSupportedMappingPath(option.value))
      .map((option) => {
        // The util encodes the real array index in the selector; read the role
        // back off the array rather than parsing it out of the display label.
        const index = Number(option.value.match(/\[(\d+)\]/)?.[1]);
        return {
          label: option.label,
          selector: option.value,
          role: found.value[index]?.role ?? '',
        };
      });
    return { messagesColumn: found.key, messageSelectors: selectors };
  }, [row]);

  const unsupportedFormat = Boolean(path) && !format;
  /** The row helpers swallow a parse failure and return null, so a resolved
   *  query is not proof the file was read. Row 0 only: a null row further in is
   *  a truncated trailing line, which ``isPartial`` already explains. */
  const unreadable = rowIndex === 0 && preview !== undefined && preview?.row === null;

  return {
    row,
    rowCount: preview?.rowCount ?? 0,
    isPartial: !isComplete,
    keyOptions,
    messagesColumn,
    messageSelectors,
    isLoading: isFetching,
    error: unsupportedFormat
      ? 'Unsupported file type. Pick a .json, .jsonl, or .csv file.'
      : error || unreadable
        ? 'Could not read the selected file.'
        : null,
  };
}
