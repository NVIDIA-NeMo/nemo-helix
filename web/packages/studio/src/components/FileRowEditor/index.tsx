// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Button, Stack } from '@nvidia/foundations-react-core';
import { Trash } from '@nvidia/react-gui-icons/icons';
import { makeDataFileColumns } from '@studio/components/FileRowEditor/columns';
import { COLUMN_PINNING, DEFAULT_SORT } from '@studio/components/FileRowEditor/constants';
import { FileHeader } from '@studio/components/FileRowEditor/FileHeader';
import {
  formatFromFileName,
  parseDataFile,
  serializeDataFile,
  TEXT_PARSEABLE_FORMATS,
  type DataFileFormat,
} from '@studio/components/FileRowEditor/parse';
import { RowEditorPanel } from '@studio/components/FileRowEditor/RowEditorPanel';
import {
  assignRowIds,
  cloneRow,
  deriveRows,
  emptyRow,
  formatBytes,
  nextId,
  rowId,
} from '@studio/components/FileRowEditor/rows';
import { inferColumns } from '@studio/components/FileRowEditor/schema';
import {
  ROW_ID_KEY,
  type DataFileColumn,
  type DataFileRow,
} from '@studio/components/FileRowEditor/types';
import {
  type ChangeEvent,
  type FC,
  type ReactNode,
  useCallback,
  useMemo,
  useRef,
  useState,
} from 'react';

export interface FileRowEditorProps {
  /** File name shown in the header. Its extension drives the format chip. */
  fileName?: string;
  /**
   * Replaces the header's static file name with a custom node (e.g. a file picker), letting
   * the header double as the file selector. The format chip and stats still track `fileName`.
   */
  slotFileName?: ReactNode;
  /** File size label shown in the header summary. */
  fileSizeLabel?: string;
  /**
   * Column schema. When omitted, it is inferred from {@link initialRows} — this is the
   * common path. Provide it to control order/labels/types or to support an empty file.
   */
  columns?: DataFileColumn[];
  /** Initial dataset rows of any row-like shape. */
  initialRows?: DataFileRow[];
  /**
   * Whether to show the "Open File" action that loads a local file in-browser. Disable
   * when rows are supplied by the host (e.g. fetched from the Files API) so the user can't
   * swap in an unrelated local file. @defaultValue true
   */
  showOpenFile?: boolean;
  /**
   * Persists the current rows to the backing store. When provided, the header shows a
   * "Save File" action (enabled only when there are unsaved changes) and edits become
   * durable. When omitted, the editor stays in-memory only. The returned promise must
   * reject on failure so the editor keeps its dirty state for a retry.
   */
  onSaveFile?: (rows: DataFileRow[]) => Promise<void> | void;
  /** Whether a save is in flight — surfaced on the "Save File" action. */
  isSaving?: boolean;
  /**
   * When set, the "Save File" action is disabled and shows this text as its tooltip — used
   * by the host to block unsafe saves (e.g. a file too large to load in full).
   */
  saveDisabledReason?: string;
  className?: string;
}

/**
 * Data File — Row Viewer / Editor.
 *
 * A `StudioDataView` table for a structured data file (Parquet/CSV/JSON/JSONL) paired
 * with a `SidePanel` row editor. The schema is inferred from the data (or supplied via
 * `columns`), so the viewer works for any row-like file rather than one fixed shape.
 * Self-contained and presentational: it owns its row state in memory and can open a
 * local JSON/JSONL/CSV file in the browser. Wire `initialRows`/`columns` and the row
 * handlers to the Files API to go live.
 */
export const FileRowEditor: FC<FileRowEditorProps> = ({
  fileName: fileNameProp = 'qa-sft-dataset-v1.parquet',
  slotFileName,
  fileSizeLabel: fileSizeLabelProp = '4.2 MB',
  columns: columnsProp,
  initialRows = [],
  showOpenFile = true,
  onSaveFile,
  isSaving = false,
  saveDisabledReason,
  className,
}) => {
  const toast = useToast();
  const [rows, setRows] = useState<DataFileRow[]>(() => assignRowIds(initialRows));
  // Snapshot of the rows as last persisted (or as first loaded), used to detect unsaved
  // changes so the "Save File" action is enabled only when there is something to save.
  const [savedSnapshot, setSavedSnapshot] = useState<string>(() =>
    JSON.stringify(assignRowIds(initialRows))
  );
  const [columns, setColumns] = useState<DataFileColumn[]>(
    () => columnsProp ?? inferColumns(initialRows)
  );
  const [fileName, setFileName] = useState(fileNameProp);
  const [fileSizeLabel, setFileSizeLabel] = useState(fileSizeLabelProp);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [editingId, setEditingId] = useState<number | null>(null);
  const [draft, setDraft] = useState<DataFileRow | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fileFormat: DataFileFormat = useMemo(() => formatFromFileName(fileName), [fileName]);

  const currentSnapshot = useMemo(() => JSON.stringify(rows), [rows]);
  const isFileDirty = currentSnapshot !== savedSnapshot;

  const dataViewState = useStudioDataViewState({
    defaultPageSize: 10,
    defaultSort: DEFAULT_SORT,
    columnPinning: COLUMN_PINNING,
  });

  const { debouncedSearchBar, debouncedColumnFilters } = dataViewState;
  const sorting = dataViewState.sorting.state;
  const { pageIndex, pageSize } = dataViewState.pagination.state;

  const processedRows = useMemo(
    () =>
      deriveRows(rows, {
        search: debouncedSearchBar,
        columnFilters: debouncedColumnFilters,
        sorting,
      }),
    [rows, debouncedSearchBar, debouncedColumnFilters, sorting]
  );
  // The hook resets pagination to page 1 when search/filters change; clamp the slice so
  // shrinking results (e.g. after a delete) never render an out-of-range empty page.
  const lastPageIndex = Math.max(0, Math.ceil(processedRows.length / pageSize) - 1);
  const safePageIndex = Math.min(pageIndex, lastPageIndex);
  const pageRows = useMemo(
    () => processedRows.slice(safePageIndex * pageSize, safePageIndex * pageSize + pageSize),
    [processedRows, safePageIndex, pageSize]
  );

  const committedRow =
    editingId === null ? null : (rows.find((row) => rowId(row) === editingId) ?? null);
  const editingIndex = editingId === null ? -1 : rows.findIndex((row) => rowId(row) === editingId);
  const isDirty =
    !!draft && !!committedRow && JSON.stringify(draft) !== JSON.stringify(committedRow);

  const openEditorRow = useCallback((row: DataFileRow) => {
    setEditingId(rowId(row));
    setDraft(cloneRow(row));
  }, []);

  const closeEditor = useCallback(() => {
    // Keep `draft` so the close animation still shows content; the next open replaces it.
    setEditingId(null);
  }, []);

  const deleteRowById = useCallback((id: number) => {
    setRows((prev) => prev.filter((row) => rowId(row) !== id));
    setEditingId((prev) => (prev === id ? null : prev));
  }, []);

  const duplicateRow = useCallback((row: DataFileRow) => {
    setRows((prev) => {
      const index = prev.findIndex((entry) => rowId(entry) === rowId(row));
      if (index === -1) {
        return prev;
      }
      const copy = { ...cloneRow(row), [ROW_ID_KEY]: nextId(prev) };
      const next = [...prev];
      next.splice(index + 1, 0, copy);
      return next;
    });
  }, []);

  const makeColumns = useMemo(
    () =>
      makeDataFileColumns(columns, {
        onEdit: openEditorRow,
        onDuplicate: duplicateRow,
        onDelete: (row) => deleteRowById(rowId(row)),
      }),
    [columns, openEditorRow, duplicateRow, deleteRowById]
  );

  const handleFieldChange = useCallback((key: string, value: unknown) => {
    setDraft((prev) => (prev ? { ...prev, [key]: value } : prev));
  }, []);

  const handleSave = () => {
    if (!draft) {
      return;
    }
    setRows((prev) => prev.map((row) => (rowId(row) === rowId(draft) ? cloneRow(draft) : row)));
    setEditingId(null);
    toast.success(onSaveFile ? 'Change applied — click Save File to persist' : 'Row saved');
  };

  const handleSaveFile = async () => {
    if (!onSaveFile || !isFileDirty || isSaving) {
      return;
    }
    const persisted = currentSnapshot;
    try {
      await onSaveFile(rows);
      setSavedSnapshot(persisted);
    } catch {
      // The host surfaces the failure; keep the dirty state so the user can retry.
    }
  };

  const handleAddRow = () => {
    const created = emptyRow(nextId(rows), columns);
    setRows((prev) => [...prev, created]);
    openEditorRow(created);
  };

  const navigateEditor = (delta: number) => {
    const target = rows[editingIndex + delta];
    if (target) {
      openEditorRow(target);
    }
  };

  const handleOpenFileClick = () => fileInputRef.current?.click();

  const handleDownload = () => {
    const downloadFormat: DataFileFormat = TEXT_PARSEABLE_FORMATS.includes(fileFormat)
      ? fileFormat
      : 'json';
    const downloadName =
      downloadFormat === fileFormat ? fileName : `${fileName.replace(/\.[^.]+$/, '')}.json`;
    const blob = new Blob([serializeDataFile(rows, downloadFormat)], {
      type: 'application/octet-stream',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = downloadName;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const handleFileSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) {
      return;
    }

    const format = formatFromFileName(file.name);
    if (!TEXT_PARSEABLE_FORMATS.includes(format)) {
      setLoadError(
        format === 'parquet'
          ? 'Parquet is binary — load it through the Files API. In-browser open supports JSON, JSONL & CSV.'
          : `Unsupported file type: ${file.name}`
      );
      return;
    }

    try {
      const text = await file.text();
      const parsed = assignRowIds(parseDataFile(text, format));
      setRows(parsed);
      setSavedSnapshot(JSON.stringify(parsed));
      setColumns(inferColumns(parsed));
      setFileName(file.name);
      setFileSizeLabel(formatBytes(file.size));
      setEditingId(null);
      setLoadError(null);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : 'Failed to parse file.');
    }
  };

  return (
    <Stack gap="density-xl" className={`h-full w-full min-w-0 ${className ?? ''}`}>
      <FileHeader
        fileName={fileName}
        slotFileName={slotFileName}
        fileFormat={fileFormat}
        rowCount={rows.length}
        columnCount={columns.length}
        fileSizeLabel={fileSizeLabel}
        loadError={loadError}
        showOpenFile={showOpenFile}
        fileInputRef={fileInputRef}
        onFileSelected={handleFileSelected}
        onOpenFileClick={handleOpenFileClick}
        onDownload={handleDownload}
        onAddRow={handleAddRow}
        downloadDisabled={rows.length === 0}
        onSaveFile={onSaveFile ? handleSaveFile : undefined}
        isSaving={isSaving}
        saveDisabled={!isFileDirty}
        saveDisabledReason={saveDisabledReason}
        hasUnsavedChanges={Boolean(onSaveFile) && isFileDirty}
      />

      {/* Table */}
      <Stack className="min-h-0 min-w-0 w-full flex-1">
        <StudioDataView<DataFileRow>
          dataViewState={dataViewState}
          makeColumns={makeColumns}
          searchField={columns.length > 0 ? columns[0].key : undefined}
          onRowClick={openEditorRow}
          renderBulkActions={({ selectedRows }) => (
            <Button
              kind="tertiary"
              color="danger"
              onClick={() => selectedRows.forEach((row) => deleteRowById(rowId(row)))}
            >
              <Trash size="16" />
              Delete ({selectedRows.length})
            </Button>
          )}
          attributes={{
            DataViewRoot: { data: pageRows, totalCount: processedRows.length },
            DataViewSearchBar: { placeholder: 'Search rows…' },
          }}
        />
      </Stack>

      <RowEditorPanel
        open={editingId !== null}
        columns={columns}
        draft={draft}
        rowNumber={editingIndex + 1}
        totalRows={rows.length}
        isDirty={isDirty}
        onFieldChange={handleFieldChange}
        onClose={closeEditor}
        onPrev={() => navigateEditor(-1)}
        onNext={() => navigateEditor(1)}
        onDelete={() => editingId !== null && deleteRowById(editingId)}
        onSave={handleSave}
      />
    </Stack>
  );
};
