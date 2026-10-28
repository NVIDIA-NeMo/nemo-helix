/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
import { isAcceptedFileType } from '@nemo/common/src/utils/file';
import { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';

/**
 * Type guard to check if a value is a FilesetFileOutput (remote file from v2 fileset API)
 */
export const isFilesetFileOutput = (
  file: FilesetFileOutput | File | undefined
): file is FilesetFileOutput => {
  return file !== undefined && 'file_ref' in file;
};

/**
 * @deprecated Use isFilesetFileOutput instead
 */
export const isListFileEntry = isFilesetFileOutput;

/**
 * Type guard to check if a value is a browser File object
 */
export const isBrowserFile = (file: FilesetFileOutput | File | undefined): file is File => {
  return file !== undefined && file instanceof File;
};

/**
 * Format bytes to human-readable file size
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'kB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
};

/**
 * If you renamed a file in a fileset, the file_ref could be duplicated. This function will return a unique id for the file.
 * @param file - The file to get the id for
 * @returns The unique id for the file
 */
export const getExistingFileId = (file: FilesetFileOutput) => {
  return `${file.file_ref}-${file.path}`;
};

/**
 * Sanitizes a filename to be a valid dataset name by:
 * - Removing file extension
 * - Replacing invalid characters with underscores
 * - Ensuring the name is not empty
 *
 * Valid dataset names must only contain alphanumeric characters, dots, underscores, and dashes.
 *
 * @param filename - The filename to sanitize
 * @returns A sanitized dataset name
 */
export const sanitizeFilenameForDatasetName = (filename: string): string => {
  // Remove file extension
  const nameWithoutExtension = filename.replace(/\.[^/.]+$/, '');

  // Replace invalid chars with underscores, then lowercase for a valid name.
  const sanitized = nameWithoutExtension.replace(/[^a-zA-Z0-9._-]/g, '_').toLowerCase();

  // If the result is empty or contains only underscores, return an empty name,
  // dataset names cannot be changed, so we return an empty name.
  if (!sanitized || /^_+$/.test(sanitized)) {
    return '';
  }

  return sanitized;
};

/**
 * UploadFile adapter over {@link isAcceptedFileType}: existing files are checked by
 * fileset path, new files by browser File name.
 */
export const isFileTypeAllowed = (
  file: { type: 'existing' | 'new'; file: { path?: string; name?: string } },
  acceptableFileTypes: string[] | undefined
): boolean =>
  isAcceptedFileType(
    (file.type === 'existing' ? file.file.path : file.file.name) ?? '',
    acceptableFileTypes
  );
