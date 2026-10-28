// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import type { IntentionalAny } from '@nemo/common/src/components/DataView/internal/types';
import { downloadFile, generateCSV } from '@nemo/common/src/components/DataView/internal/utils/csv';
import { Download } from '@nemo/icons/icons';
import { Button, type ButtonProps } from '@nvidia/foundations-react-core';
import type { Column, Row, Table } from '@tanstack/react-table';
import { Fragment, type JSX } from 'react';

export interface DownloadButtonFileContent {
  content: string;
  mimeType: string;
}

export interface PrepareDownloadContext<TData> {
  table: Table<TData>;
  rows: Row<TData>[];
  columns: Column<TData>[];
}

export interface DownloadButtonProps extends Omit<ButtonProps, 'onClick'> {
  /** The filename to use when downloading the file. @defaultValue "data.csv" */
  filename?: string;
  /**
   * Function to prepare the file content for download. Receives the table, rows, and columns
   * and should return the content string and MIME type.
   * @defaultValue Returns CSV content with "text/csv;charset=utf-8;" MIME type.
   */
  prepareDownload?: (context: PrepareDownloadContext<IntentionalAny>) => DownloadButtonFileContent;
  /** Optional callback invoked after preparing download with the rows and generated content. */
  onClick?: (data: { rows: Row<IntentionalAny>[]; content: string }) => void;
}

const defaultPrepareDownload = ({
  rows,
  columns,
}: PrepareDownloadContext<IntentionalAny>): DownloadButtonFileContent => ({
  content: generateCSV(rows, columns),
  mimeType: 'text/csv;charset=utf-8;',
});

/**
 * A button that downloads table data as a file. By default exports all visible rows as CSV.
 * Use `prepareDownload` to customize output format or row selection.
 */
export function DownloadButton({
  filename = 'data.csv',
  prepareDownload = defaultPrepareDownload,
  onClick,
  children = (
    <Fragment>
      <Download variant="fill" />
      <span className="hide-mobile">Download</span>
    </Fragment>
  ),
  ...props
}: DownloadButtonProps): JSX.Element {
  const { table } = useInnerDataViewContext();
  const handleClick = () => {
    const rows = table.getFilteredRowModel().rows;
    const columns = table.getVisibleLeafColumns();
    const { content, mimeType } = prepareDownload({ table, rows, columns });
    onClick?.({ rows, content });
    downloadFile(content, filename, mimeType);
  };
  return (
    <Button aria-label="Download data" kind="tertiary" onClick={handleClick} {...props}>
      {children}
    </Button>
  );
}
