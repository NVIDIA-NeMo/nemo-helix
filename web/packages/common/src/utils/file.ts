// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parquetRead } from 'hyparquet';
import Papa from 'papaparse';

import { FileFormatType } from '../types';

/**
 * Searches through an object's keys (one layer deep) to find an array that contains
 * at least one object with a role key in {user, assistant, system} and a non-empty content key.
 * This identifies if there is a messages array in the row.
 * @param item The object to search
 * @returns An object with the key and array value if found, null otherwise
 */
export function findMessagesArray(
  item: Record<string, unknown>
): { key: string; value: Array<{ role?: string; content?: string }> } | null {
  for (const [key, value] of Object.entries(item)) {
    // Check if value is an array
    if (!Array.isArray(value)) {
      continue;
    }

    // Check if array contains at least one object with a valid role and non-empty content
    for (const element of value) {
      // Check if element is an object
      if (typeof element !== 'object' || element === null) {
        continue;
      }

      const message = element as { role?: string; content?: string };

      // Check if role is user, assistant, or system and content is a non-empty string
      if (
        (message.role === 'user' || message.role === 'assistant' || message.role === 'system') &&
        typeof message.content === 'string' &&
        message.content.length > 0
      ) {
        return {
          key,
          value: value as Array<{ role?: string; content?: string }>,
        };
      }
    }
  }

  return null;
}

/**
 * Triggers a browser download for a file.
 * @param data - The data to be downloaded (e.g., ArrayBuffer, Blob, File).
 * @param filename - The name to give the downloaded file.
 */
export const triggerDownload = (data: BlobPart, filename: string) => {
  // Re-wrapping a Blob would drop its type, and the browser uses that to pick the extension.
  const blob = data instanceof Blob ? data : new Blob([data]);

  // Create an object URL for the Blob
  const blobUrl = URL.createObjectURL(blob);

  // Create a temporary anchor element
  const a = document.createElement('a');

  try {
    a.href = blobUrl;
    a.download = filename;

    // Hide the element and add it to the DOM
    a.style.display = 'none';
    document.body.appendChild(a);

    // Programmatically click the element to trigger the download
    a.click();
  } finally {
    // Clean up by removing the element and revoking the URL
    document.body.removeChild(a);
    URL.revokeObjectURL(blobUrl);
  }
};

/**
 * Extracts the first row from a file based on its format.
 * Convenience wrapper around getRowAtIndex for index 0.
 * @param file The file to read
 * @param format The file format ('json' or 'jsonl')
 * @returns The first row object or null if empty/invalid
 */
export async function getFirstRow(
  file: File,
  format: FileFormatType
): Promise<Record<string, unknown> | null> {
  return getRowAtIndex(file, format, 0);
}

/**
 * Gets the total number of rows in a file.
 * For JSONL files, counts newlines without parsing individual objects.
 * For JSON files, parses the file to determine if it's an array (returns length) or object (returns 1).
 * @param file - The file to count rows from
 * @param format - The file format ('json' or 'jsonl')
 * @returns The total number of rows, or 0 if file cannot be read
 */
export async function getFileRowCount(file: File, format: FileFormatType): Promise<number> {
  try {
    if (format === 'parquet') {
      let count = 0;
      const buffer = await file.arrayBuffer();
      await parquetRead({
        file: buffer,
        rowFormat: 'object',
        onComplete: (rows) => {
          count = rows.length;
        },
      });
      return count;
    }

    const text = await file.text();

    if (format === 'jsonl') {
      const lines = text
        .trim()
        .split('\n')
        .filter((line) => line.length > 0);
      return lines.length;
    } else if (format === 'csv') {
      const result = Papa.parse<Record<string, string>>(text, {
        header: true,
        skipEmptyLines: true,
      });
      return result.data.length;
    } else {
      const data = JSON.parse(text);
      return Array.isArray(data) ? data.length : 1;
    }
  } catch {
    return 0;
  }
}

/**
 * Fetches a specific row from a file by index.
 * For large files, this still requires reading the entire file, but only parses the needed row.
 * @param file - The file to read from
 * @param format - The file format ('json' or 'jsonl')
 * @param index - The zero-based index of the row to fetch
 * @returns The row data at the specified index, or null if not found
 */
export async function getRowAtIndex(
  file: File,
  format: FileFormatType,
  index: number
): Promise<Record<string, unknown> | null> {
  try {
    if (format === 'parquet') {
      let targetRow: Record<string, unknown> | null = null;
      const buffer = await file.arrayBuffer();
      await parquetRead({
        file: buffer,
        rowFormat: 'object',
        rowStart: index,
        rowEnd: index + 1,
        onComplete: (rows) => {
          const row = rows[0];
          if (row) {
            // Coerce BigInt (from INT64 columns) to number for JSON Schema inference.
            targetRow = Object.fromEntries(
              Object.entries(row).map(([k, v]) => [k, typeof v === 'bigint' ? Number(v) : v])
            );
          }
        },
      });
      return targetRow;
    }

    const text = await file.text();

    if (format === 'jsonl') {
      const lines = text
        .trim()
        .split('\n')
        .filter((line) => line.length > 0);
      if (index >= 0 && index < lines.length) {
        return JSON.parse(lines[index]);
      }
      return null;
    } else if (format === 'csv') {
      const result = Papa.parse<Record<string, string>>(text, {
        header: true,
        skipEmptyLines: true,
      });
      return (result.data[index] as Record<string, unknown>) ?? null;
    } else {
      const data = JSON.parse(text);
      if (Array.isArray(data)) {
        return data[index] ?? null;
      }
      // For non-array JSON, only index 0 is valid
      return index === 0 ? data : null;
    }
  } catch {
    return null;
  }
}

/**
 * Extracts all available keys from a data row for use in dropdown selections.
 * For regular keys, returns them with the key name as both label and value.
 * For messages arrays, expands to show individual message content paths with descriptive labels.
 * @param item The data row object to extract keys from
 * @param messagesArrayResult Optional pre-computed result from findMessagesArray to avoid redundant computation
 * @returns An array of objects with label and value properties for dropdown options
 */
export function extractUserFriendlyKeysFromRow(
  item: Record<string, unknown>,
  messagesArrayResult?: { key: string; value: Array<{ role?: string; content?: string }> } | null
): Array<{ label: string; value: string }> {
  const keys: Array<{ label: string; value: string }> = [];

  // Use provided result or compute it
  const result = messagesArrayResult ?? findMessagesArray(item);
  const messagesKey = result?.key;
  const messagesArray = result?.value;

  // Iterate through all keys in the object
  for (const [key, value] of Object.entries(item)) {
    // If this is the messages array, expand it to show individual message content paths
    if (key === messagesKey && Array.isArray(value) && messagesArray) {
      value.forEach((message, index) => {
        // Only include messages that have content
        if (
          typeof message === 'object' &&
          message !== null &&
          'content' in message &&
          typeof message.content === 'string' &&
          message.content.length > 0 &&
          'role' in message &&
          typeof message.role === 'string'
        ) {
          const selector = `${key}[${index}].content`;
          const label = `${key}: ${message.role} message`;
          keys.push({ label, value: selector });
        }
      });
    } else {
      // Regular key - include with key name as both label and value
      keys.push({ label: key, value: key });
    }
  }

  return keys;
}

/**
 * Resolves a dot-bracket path (e.g. "messages[0].content") against a row object.
 * These paths match the key format produced by extractUserFriendlyKeysFromRow.
 * A predicate segment such as "messages[role=assistant]" selects the last array
 * element whose field equals that string, as the evaluator's field_mapping does.
 * @param row The data row object
 * @param path The dot-bracket path to resolve
 * @returns The value at that path, or undefined if the path doesn't resolve
 */
export function resolveKeyPath(row: Record<string, unknown>, path: string): unknown {
  const segments = path
    .replace(/\[(\d+)\]/g, '.$1')
    .replace(/\[([^[\]=.]+=[^[\]=.]+)\]/g, '.[$1]')
    .split('.');
  let current: unknown = row;
  for (const segment of segments) {
    if (current === null || current === undefined || typeof current !== 'object') {
      return undefined;
    }
    const predicate = /^\[(.+)=(.+)\]$/.exec(segment);
    if (predicate) {
      if (!Array.isArray(current)) return undefined;
      const [, key, literal] = predicate;
      current = current.findLast(
        (item: unknown) =>
          typeof item === 'object' &&
          item !== null &&
          (item as Record<string, unknown>)[key] === literal
      );
      if (current === undefined) return undefined;
      continue;
    }
    if (!Object.prototype.hasOwnProperty.call(current, segment)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[segment];
  }
  return current;
}
