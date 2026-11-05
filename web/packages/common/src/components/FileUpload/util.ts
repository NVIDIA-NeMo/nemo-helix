// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Accept, ErrorCode, type FileRejection } from 'react-dropzone';

/**
 * @returns true if any of the errors in any of the file rejections is for too many files
 */
export const hadTooManyFilesError = (fileRejections: FileRejection[]) =>
  fileRejections.some((r) => r.errors.some((e) => e.code === ErrorCode.TooManyFiles));

/**
 * @returns true if any of the errors in any of the file rejections is for wrong file type
 */
export const hadInvalidFileTypeError = (fileRejections: FileRejection[]) =>
  fileRejections.some((r) => r.errors.some((e) => e.code === ErrorCode.FileInvalidType));

/**
 * @returns an array of all the accepted file extensions across all mime types
 */
export const getAcceptedFileExtensions = (accept: Accept) => {
  return Object.values(accept).flat();
};
