// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';

export interface FilesetPickerNoun {
  readonly one: string;
  readonly many: string;
}

export const DEFAULT_FILESET_PICKER_NOUN: FilesetPickerNoun = { one: 'fileset', many: 'filesets' };

export type FilesetPickerPurpose = FilesetPurpose | readonly FilesetPurpose[];

export type FilesetSelectionError = 'missing' | 'unavailable';
