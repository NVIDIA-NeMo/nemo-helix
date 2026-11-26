// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FilesetPickerCore,
  type FilesetPickerCoreProps,
} from '@nemo/common/src/components/FilesetPicker/FilesetPickerCore';
import { filesListFilesets, filesRetrieveFileset } from '@nemo/sdk/generated/platform/files';
import type { FC } from 'react';

export type FilesetPickerProps = Omit<FilesetPickerCoreProps, 'listFilesets' | 'retrieveFileset'>;

/** `FilesetPickerCore` on Studio's SDK client. */
export const FilesetPicker: FC<FilesetPickerProps> = (props) => (
  <FilesetPickerCore
    {...props}
    listFilesets={filesListFilesets}
    retrieveFileset={filesRetrieveFileset}
  />
);
