// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { logger } from '@nemo/common/src/utils/logger';
import { FileSystemDirectory, FileSystemNode } from '@studio/components/FilesTable/utils';
import { parseCSVTable } from '@studio/components/SafeSynthesizerFilesetPreview/util';
import { getTextWithCount, parseCSV } from '@studio/util/strings';

/**
 * @returns a string with a human-readable file size
 */
export const getHumanReadableFileSize = (size: number, base: number = 1024) => {
  if (size === 0) {
    return 'empty file';
  }
  const units = ['B', 'kB', 'MB', 'GB', 'TB', 'PB'];
  const exponent = Math.floor(Math.log(size) / Math.log(base));
  const numUnits =
    exponent >= units.length
      ? size / Math.pow(base, units.length - 1)
      : size / Math.pow(base, exponent);

  return Number(numUnits.toFixed(2)).toLocaleString() + units[Math.min(exponent, units.length - 1)];
};

export const getTextSizeInBytes = (text: string): number => {
  return new TextEncoder().encode(text).length;
};

/**
 * Returns the file name from a path. If not possible returns the path.
 * Example:
 *   /training/training_file.jsonl -> training_file.jsonl
 *   foobar123 -> foobar123
 */
export const getFileNameFromPath = (path: string) => {
  return path.match(/[^/\\]+$/)?.[0] ?? path;
};

/**
 * Returns the given fileName prepended with the given folder, if provided.
 */
export const getFullFilePath = (filepath: string, folder?: string) => {
  const fileName = getFileNameFromPath(filepath);
  return folder ? `${folder}/${fileName}` : fileName;
};

/**
 * Resolves the storage path for a file within a dataset for API calls (download, delete, etc.).
 * Tree nodes and API responses use full paths (e.g. `training/data/file.txt`). When the path
 * already contains `/`, it is returned as-is. Only single-segment paths (filename or root-level
 * name) are joined with `folder` — the legacy case for folder-scoped listings.
 */
export const resolveDatasetFilePath = (filepath: string, folder?: string) => {
  if (filepath.includes('/')) {
    return filepath;
  }
  if (folder) {
    const normalized = folder.endsWith('/') ? folder.slice(0, -1) : folder;
    return `${normalized}/${filepath}`;
  }
  return filepath;
};

/**
 * Collects unique folder prefixes from dataset file paths (sorted), for folder pickers.
 */
export const collectFolderPathsFromDatasetFiles = (
  files: ReadonlyArray<{ path: string }> | undefined
): string[] => {
  if (!files?.length) return [];
  const folders = new Set<string>();
  for (const f of files) {
    const parts = f.path.split('/');
    let acc = '';
    for (let i = 0; i < parts.length - 1; i++) {
      acc = acc ? `${acc}/${parts[i]}` : parts[i];
      folders.add(acc);
    }
  }
  return Array.from(folders).sort((a, b) => a.localeCompare(b));
};

export const renameFile = (file: File, newName: string): File => {
  const blob = file.slice(0, file.size, file.type);
  return new File([blob], newName, {
    type: file.type,
    lastModified: file.lastModified,
  });
};

interface ParseFileContentInput {
  content: string;
  fileType?: string;
}

export type Row = Record<string, unknown>;

export interface ParseFileContentReturn {
  rows: Row[];
  failures?: string[];
}
/**
 * Given a string in {@link ALLOWED_CONTENT_FILE_TYPES}, parse the content to return a
 * list of objects (rows) and failures.
 * @param content - File contents represented as a string.
 * @param fileType - The file extension.
 */
export const parseFileContent = ({
  content,
  fileType,
}: ParseFileContentInput): ParseFileContentReturn => {
  try {
    if (fileType?.includes('csv')) {
      const csvData = parseCSV({
        csvString: content,
        options: { header: true, skipEmptyLines: true },
      });
      return { rows: csvData };
    }
    const jsonData = JSON.parse(content);
    if (Array.isArray(jsonData)) {
      return { rows: jsonData };
    } else {
      return { rows: [jsonData] };
    }
  } catch {
    const failures: string[] = [];
    // It's misformatted, OR JSONL or Parquet
    try {
      const rows = content
        .split('\n')
        .filter((line) => line.trim() !== '')
        .reduce(
          (rows, row) => {
            try {
              rows.push(JSON.parse(row));
              return rows;
            } catch {
              failures.push(row);
              logger.warn(`Invalid JSON row ignored: ${row}`);
              return rows;
            }
          },
          [] as Record<string, unknown>[]
        );
      return { rows, failures };
    } catch {
      throw new Error(`Error while parsing file content.`);
    }
  }
};

interface GetContentSchemaReturn {
  schema: Record<string, string>;
  total_rows: number;
}
interface ContentSchemaOptions {
  all?: boolean;
  fileType?: string;
}
/**
 * Given a content string representing a file, handle parsing the row(s) of the file
 * to get a schema repsentation.
 * TODO: Handle nested object schema types
 * @param content - File content.
 * @param opts.oneRow - Parse only one row. Efficient at the cost of evaluating the whole file.
 */
export const getContentSchema = (content?: string, opts?: ContentSchemaOptions) => {
  if (!content) return {} as Partial<GetContentSchemaReturn>;
  const { rows } = parseFileContent({ content, fileType: opts?.fileType });
  const ret: GetContentSchemaReturn = { schema: {}, total_rows: rows.length };
  const parseRow = (row: object) => {
    if (typeof row === 'object') {
      for (const [key, value] of Object.entries(row)) {
        const valueType = Array.isArray(value) ? 'array' : typeof value;

        // Add or ensure consistency in the schema
        if (!ret.schema[key]) {
          ret.schema[key] = valueType;
        } else if (ret.schema[key] !== valueType) {
          ret.schema[key] = 'mixed'; // Handle inconsistent types
        }
      }
    }
  };

  if (opts?.all) {
    for (const row of rows) {
      parseRow(row);
    }
  } else {
    parseRow(rows[0]);
  }

  return ret;
};

export const getContentColumns = (content?: string, fileType?: string): string[] => {
  if (!content) return [];
  if (fileType?.includes('csv')) {
    return parseCSVTable(content).columns.map((column) => column.children);
  }
  const { rows } = parseFileContent({ content, fileType });
  const firstRow = rows[0];
  return firstRow && typeof firstRow === 'object' ? Object.keys(firstRow) : [];
};

/**
 * Infers the JSON content type based on file extension.
 *
 * @param filePath - The file path to analyze
 * @returns ContentType.JSON for .json files, ContentType.JSONL for .jsonl files, null for all other files
 *
 * @example
 * inferJsonContentType('data.json') // ContentType.JSON
 * inferJsonContentType('data.jsonl') // ContentType.JSONL
 * inferJsonContentType('data.txt') // null
 * inferJsonContentType('myfile') // null
 * inferJsonContentType('file.tar.gz') // null
 */
export const inferJsonContentType = (filePath: string): ContentType | null => {
  const parts = filePath.toLowerCase().split('.');

  // If there's no dot, it's a file without extension
  if (parts.length === 1) {
    return null;
  }

  const extension = parts[parts.length - 1];
  if (extension === 'jsonl') {
    return ContentType.JSONL;
  }
  if (extension === 'json') {
    return ContentType.JSON;
  }
  return null;
};

/**
 * Checks if a content type represents a JSON file type.
 *
 * @param contentType - The content type to check
 * @returns true if the content type is JSON or JSONL, false otherwise
 *
 * @example
 * isJsonFile(ContentType.JSON) // true
 * isJsonFile(ContentType.JSONL) // true
 * isJsonFile(null) // false
 */
export const isJsonFile = (contentType: string | null): boolean => {
  return contentType === ContentType.JSON || contentType === ContentType.JSONL;
};

/**
 * Extracts a human-readable display name from a dataset files_url.
 * For HuggingFace URLs like "hf://datasets/username/dataset-name/file.jsonl",
 * returns "dataset-name/file.jsonl" (the last two path segments).
 *
 * @param filesUrl The files_url from a dataset (e.g., "hf://datasets/odrulea/cooperative-hedgehog/evaluation.jsonl")
 * @returns A human-friendly display name (e.g., "cooperative-hedgehog/evaluation.jsonl"), or undefined if input is falsy
 */
export const getDatasetDisplayNameFromFilesUrl = (
  filesUrl: string | undefined
): string | undefined => {
  // Handle undefined/null/empty input
  if (!filesUrl) return undefined;

  // Remove the protocol prefix (hf://, nds:, etc.)
  const url = filesUrl.replace(/^[a-z]+:\/\/|^[a-z]+:/, '');
  const parts = url.split('/');

  // For HuggingFace URLs: hf://datasets/username/dataset-name/file.jsonl
  // We want the last two parts: "dataset-name/file.jsonl"
  if (parts.length >= 4 && parts[0] === 'datasets') {
    return parts.slice(-2).join('/');
  }

  // For other URL formats, return the last two segments if available
  if (parts.length >= 2) {
    return parts.slice(-2).join('/');
  }

  // Fallback to the original URL
  return filesUrl;
};

export const getFolderSize = (folder: FileSystemDirectory): string => {
  let totalSize = 0;
  let numFiles = 0;

  const calculateRecursive = (node: FileSystemNode) => {
    if (node.type === 'file') {
      totalSize += node.size;
      numFiles += 1;
    } else if (node.type === 'directory') {
      Object.values(node.children).forEach(calculateRecursive);
    }
  };

  Object.values(folder.children).forEach(calculateRecursive);

  return `${getHumanReadableFileSize(totalSize)} (${getTextWithCount('file', numFiles)})`;
};
