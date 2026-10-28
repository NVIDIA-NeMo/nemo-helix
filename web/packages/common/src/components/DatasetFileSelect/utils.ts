// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { FileListItem } from '@nemo/common/src/components/FileList';

/**
 * Parse fileset URL format (fileset://workspace/name/filepath) into a FileListItem.
 * Note: dataset field is not included since we only have workspace/name from the URL.
 * The caller should fetch the full FilesetOutput if needed.
 */
export const fromFilesetUrl = (url: string): FileListItem | null => {
  const parsed = parseFilesetUrl(url);
  if (!parsed) {
    return null;
  }

  return {
    path: parsed.path,
    url,
  };
};

/**
 * Infers the JSON content type based on file extension.
 */
export const inferJsonContentType = (filePath: string): ContentType | null => {
  const parts = filePath.toLowerCase().split('.');

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
 */
export const isJsonFile = (contentType: string | null): boolean => {
  return contentType === ContentType.JSON || contentType === ContentType.JSONL;
};

/**
 * Parse `fileset://` URLs into workspace, fileset name, and object path.
 * Uses {@link parseFilesetLocation} so hash and slash forms match the Python SDK / job artifacts.
 *
 * @returns `null` if the URL is not `fileset://`, or if there is no path inside the fileset (root-only refs).
 */
export const parseFilesetUrl = (
  url: string
): { workspace: string; name: string; path: string } | null => {
  const trimmed = url.trim();
  if (!trimmed.startsWith('fileset://') && trimmed.includes(':')) {
    return null;
  }
  const loc = parseFilesetLocation(trimmed);
  if (!loc?.objectPath) {
    return null;
  }
  return { workspace: loc.workspace, name: loc.name, path: loc.objectPath };
};
