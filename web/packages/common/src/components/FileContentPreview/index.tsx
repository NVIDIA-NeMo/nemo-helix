// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import {
  inferJsonContentType,
  isJsonFile,
} from '@nemo/common/src/components/DatasetFileSelect/utils';
import type { FileListItem } from '@nemo/common/src/components/FileList';
import { MarkdownContent } from '@nemo/common/src/components/MarkdownContent';
import { ScrollTable } from '@nemo/common/src/components/ScrollTable';
import { getFileExtension } from '@nemo/common/src/utils/file';
import { Flex, Spinner, TableRowDefinition, Text } from '@nvidia/foundations-react-core';
import Papa from 'papaparse';
import { type FC, lazy, Suspense, useEffect, useMemo, useState } from 'react';

const MARKDOWN_EXTENSIONS = new Set(['.md', '.markdown']);

const CodeEditor = lazy(() =>
  import('@nemo/common/src/components/CodeEditor').then((m) => ({ default: m.CodeEditor }))
);

const editorFallback = (
  <Flex align="center" justify="center" className="h-full">
    <Spinner size="medium" aria-label="Loading editor..." />
  </Flex>
);

export interface FileContentPreviewProps {
  isLoading: boolean;
  error: Error | null;
  content?: string;
  file: FileListItem;
}

export const FileContentPreview: FC<FileContentPreviewProps> = ({
  isLoading,
  error,
  content,
  file,
}) => {
  const [parseError, setParseError] = useState<string | undefined>(undefined);
  const [csvData, setCsvData] = useState<{
    data: Record<string, unknown>[];
    columns: string[];
  } | null>(null);

  // Detect file types
  const jsonContentType = useMemo(() => inferJsonContentType(file.path), [file.path]);
  const isJson = useMemo(() => isJsonFile(jsonContentType), [jsonContentType]);
  const extension = useMemo(() => getFileExtension(file.path), [file.path]);
  const isCsv = extension === '.csv';
  const isMarkdown = extension !== null && MARKDOWN_EXTENSIONS.has(extension);

  // Parse CSV if applicable
  useEffect(() => {
    if (!isCsv || !content || isLoading) {
      setCsvData(null);
      setParseError(undefined);
      return;
    }

    const parsed = Papa.parse(content, {
      header: true,
      skipEmptyLines: true,
    });

    if (parsed.errors.length > 0) {
      const errorMessage = parsed.errors[0].message;
      setParseError(errorMessage);
      setCsvData(null);
      return;
    }

    const data = parsed.data as Record<string, unknown>[];
    const columns = parsed.meta.fields || [];
    setCsvData({ data, columns });
    setParseError(undefined);
  }, [isCsv, content, isLoading]);

  if (isLoading) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Spinner size="medium" aria-label="Loading..." />
      </Flex>
    );
  }

  if (error) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Text className="text-danger-base">
          Error: {error?.message ? error.message : 'Failed to load file'}
        </Text>
      </Flex>
    );
  }

  if (parseError) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Text className="text-danger-base">Error: {parseError}</Text>
      </Flex>
    );
  }

  if (!content) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Text>No content available</Text>
      </Flex>
    );
  }

  // Markdown files - rendered
  if (isMarkdown) {
    return (
      <div className="h-full overflow-auto rounded-lg border border-base bg-surface-raised p-4">
        <MarkdownContent content={content} />
      </div>
    );
  }

  // JSON / JSONL files
  if (isJson && jsonContentType) {
    return (
      <div className="h-full min-h-0">
        <Suspense fallback={editorFallback}>
          <CodeEditor
            content={content}
            contentType={jsonContentType}
            readOnly
            className="h-full min-h-0"
          />
        </Suspense>
      </div>
    );
  }

  // CSV files - use ScrollTable
  if (isCsv && csvData) {
    const columns = csvData.columns.map((column: string) => ({
      children: column,
    }));

    const rows: TableRowDefinition[] = csvData.data.map((row, index) => ({
      id: index.toString(),
      cells: csvData.columns.map((column: string) => ({
        children: String(row[column] ?? ''),
      })),
    }));

    return (
      <div className="h-full p-4">
        <ScrollTable columns={columns} rows={rows} pagination={false} allowHorizontalScroll />
      </div>
    );
  }

  // Plain text fallback (incl. .txt, .log, anything we don't have a richer view for)
  return (
    <div className="h-full min-h-0">
      <Suspense fallback={editorFallback}>
        <CodeEditor
          content={content}
          contentType={ContentType.TEXT}
          readOnly
          className="h-full min-h-0"
        />
      </Suspense>
    </div>
  );
};
