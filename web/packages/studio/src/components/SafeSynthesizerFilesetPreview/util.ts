// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Papa from 'papaparse';
import type { ReactNode } from 'react';

export interface ParsedCSVTable {
  rows: { id: string; cells: { children: ReactNode }[] }[];
  columns: { children: string }[];
}

export const parseCSVTable = (response: string): ParsedCSVTable => {
  const csvData = Papa.parse(response, { header: true });
  const rawRows = csvData.data as Record<string, unknown>[];
  const columnNames = csvData.meta.fields || [];

  // Format columns for ScrollTable
  const columns = columnNames.map((column) => ({
    children: column,
  }));

  // Format rows for ScrollTable
  const rows = rawRows.map((row, index) => ({
    id: row.id ? String(row.id) : String(index),
    cells: columnNames.map((column) => ({
      children: (row[column] ?? '') as ReactNode,
    })),
  }));

  return { rows, columns };
};

export interface ParsedFileContent {
  type: 'json' | 'csv' | 'error';
  jsonData?: string;
  tabularData?: ParsedCSVTable;
  error?: string;
}

/**
 * Parses file content based on file extension
 * @param filePath - The path/name of the file (used to determine type)
 * @param content - The file content as a string
 * @returns Parsed file content or error
 */
export const parseFileContent = (filePath: string, content: string): ParsedFileContent => {
  if (filePath.endsWith('.csv')) {
    const csvData = parseCSVTable(content);
    return { type: 'csv', tabularData: csvData };
  }
  if (filePath.endsWith('.json') || filePath.endsWith('.jsonl')) {
    return { type: 'json', jsonData: content };
  }
  return { type: 'error', error: 'Unsupported file type' };
};
