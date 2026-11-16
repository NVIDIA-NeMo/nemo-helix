// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FormFieldProps } from '@nvidia/foundations-react-core';
import type { UseControllerProps } from 'react-hook-form';
import { z } from 'zod';

export type ResourceRef = `${string}/${string}`;
/** @deprecated Use ResourceRef for workspace/name references. */
export type URN = ResourceRef;
export const resourceRefRegExp = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;
export const resourceRefSchema = z
  .string()
  .regex(resourceRefRegExp, 'Must be in the form "workspace/name"') as z.ZodType<ResourceRef>;

export interface UseControllerComponentProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  useControllerProps: UseControllerProps<any>;
  formFieldProps?: FormFieldProps;
}

/**
 * Enum for supported file formats
 */
export enum FileFormat {
  JSON = 'json',
  JSONL = 'jsonl',
  CSV = 'csv',
  PARQUET = 'parquet',
}

export const SUPPORTED_FILE_FORMATS: FileFormat[] = [
  FileFormat.JSON,
  FileFormat.JSONL,
  FileFormat.CSV,
  FileFormat.PARQUET,
];

/**
 * Enum for supported input file schema types
 */
export enum InputFileSchemaType {
  CHAT_COMPLETION = 'chat-completion',
  COMPLETION = 'completion',
}

/**
 * Type for file format detection
 */
export type FileFormatType = `${FileFormat}`;
