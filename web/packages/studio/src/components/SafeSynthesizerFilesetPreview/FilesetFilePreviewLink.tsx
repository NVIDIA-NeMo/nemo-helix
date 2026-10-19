// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetUrl } from '@nemo/common/src/components/DatasetFileSelect/utils';
import { ScrollTable } from '@nemo/common/src/components/ScrollTable';
import { useFilesDownloadFile } from '@nemo/sdk/generated/platform/files';
import {
  Button,
  CodeSnippet,
  type TableColumnDefinition,
  type TableRowDefinition,
} from '@nvidia/foundations-react-core';
import { FilePreview } from '@studio/components/SafeSynthesizerFilesetPreview/FilePreview';
import { parseFileContent } from '@studio/components/SafeSynthesizerFilesetPreview/util';
import { FC, ReactNode, useCallback, useMemo, useState } from 'react';

interface FilesetFilePreviewLinkProps {
  /** Fileset URL (e.g., fileset://workspace/name/path/to/file.csv) */
  url: string;
  /** Content to render inside the anchor - if not provided, displays the formatted path */
  children?: ReactNode;
}

/**
 * A component that renders a clickable link for fileset files.
 * When clicked, opens a preview panel showing the file contents.
 * Supports CSV, JSON, and JSONL file formats.
 */
export const FilesetFilePreviewLink: FC<FilesetFilePreviewLinkProps> = ({ url, children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const [showPreviewPanel, setShowPreviewPanel] = useState(false);
  const [previewTitle, setPreviewTitle] = useState<string>('');
  const [jsonDataPreview, setJsonDataPreview] = useState<string | undefined>(undefined);
  const [tabularDataPreview, setTabularDataPreview] = useState<
    | {
        rows: TableRowDefinition[];
        columns: TableColumnDefinition[];
      }
    | undefined
  >(undefined);

  const filesetInfo = useMemo(() => (url ? parseFilesetUrl(url) : null), [url]);

  const { refetch: refetchFile } = useFilesDownloadFile(
    filesetInfo?.workspace ?? '',
    filesetInfo?.name ?? '',
    filesetInfo?.path ?? '',
    { query: { enabled: false } }
  );

  const handlePreviewFile = useCallback((filePath: string, content: string) => {
    const parsed = parseFileContent(filePath, content);

    if (parsed.type === 'csv' && parsed.tabularData) {
      setTabularDataPreview(parsed.tabularData);
    } else if (parsed.type === 'json' && parsed.jsonData) {
      setJsonDataPreview(parsed.jsonData);
    } else if (parsed.type === 'error') {
      setError(parsed.error);
    }
  }, []);

  const handlePreviewClick = useCallback(async () => {
    if (!filesetInfo) return;

    setIsLoading(true);
    setShowPreviewPanel(true);
    setError(undefined);
    setJsonDataPreview(undefined);
    setTabularDataPreview(undefined);
    setPreviewTitle(`${filesetInfo.workspace}/${filesetInfo.name}/${filesetInfo.path}`);

    try {
      const response = await refetchFile();
      if (!response.data) {
        setError('Error fetching file');
        return;
      }
      const content = await response.data.text();
      handlePreviewFile(filesetInfo.path, content);
    } catch {
      setError('Could not parse file');
    } finally {
      setIsLoading(false);
    }
  }, [filesetInfo, refetchFile, handlePreviewFile]);

  const handleClosePanel = useCallback(() => {
    setError(undefined);
    setShowPreviewPanel(false);
    setJsonDataPreview(undefined);
    setTabularDataPreview(undefined);
  }, []);

  const handleDownload = useCallback(async () => {
    try {
      const response = await refetchFile();
      // SDK returns a Blob directly
      return response.data ?? null;
    } catch {
      return null;
    }
  }, [refetchFile]);

  return (
    <>
      {showPreviewPanel && (
        <FilePreview
          error={error}
          isLoading={isLoading}
          onClose={handleClosePanel}
          title={previewTitle}
          onDownload={handleDownload}
          downloadFileName={filesetInfo?.path ?? ''}
        >
          {tabularDataPreview && (
            <ScrollTable
              allowHorizontalScroll
              columns={tabularDataPreview.columns}
              rows={tabularDataPreview.rows}
            />
          )}
          {jsonDataPreview && (
            <CodeSnippet
              language="json"
              kind="block"
              attributes={{ CodeSnippetCode: { className: '[&&]:min-h-auto' } }}
              value={jsonDataPreview}
            />
          )}
        </FilePreview>
      )}

      <Button
        kind="tertiary"
        onClick={handlePreviewClick}
        className="truncate max-w-full text-left p-0 h-auto"
      >
        {children}
      </Button>
    </>
  );
};
