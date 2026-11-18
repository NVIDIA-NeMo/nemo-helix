// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import {
  getFilesetDetailRoute,
  getWorkspaceFilesetsRoute,
  type FilesetDetailRouteOptions,
} from '@studio/routes/utils';
import { createContext, useContext } from 'react';

/**
 * The list page a fileset detail page sits under. It sets the breadcrumb, where the page
 * goes after a delete, and the links the page builds back to itself (folders, files).
 */
export interface FilesetDetailParent {
  label: string;
  /** Only filesets of this purpose belong here; the page sends others to Filesets. */
  purpose?: FilesetPurpose;
  getListRoute: (workspace: string) => string;
  getDetailRoute: (
    workspace: string,
    filesetName: string,
    options?: FilesetDetailRouteOptions
  ) => string;
}

export const FILESETS_PARENT: FilesetDetailParent = {
  label: 'Filesets',
  getListRoute: getWorkspaceFilesetsRoute,
  getDetailRoute: getFilesetDetailRoute,
};

export const FilesetDetailParentContext = createContext<FilesetDetailParent>(FILESETS_PARENT);

/** Outside a detail page this falls back to the Filesets section. */
export const useFilesetDetailParent = () => useContext(FilesetDetailParentContext);
