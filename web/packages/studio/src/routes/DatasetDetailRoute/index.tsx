// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetDetailRoute } from '@studio/routes/FilesetDetailRoute';
import type { FilesetDetailParent } from '@studio/routes/FilesetDetailRoute/FilesetDetailParent';
import { getDatasetDetailRoute, getWorkspaceDatasetsRoute } from '@studio/routes/utils';
import type { FC } from 'react';

const DATASETS_PARENT: FilesetDetailParent = {
  label: 'Datasets',
  getListRoute: getWorkspaceDatasetsRoute,
  getDetailRoute: getDatasetDetailRoute,
};

/** The fileset detail page, with breadcrumbs and links kept under /datasets. */
export const DatasetDetailRoute: FC = () => <FilesetDetailRoute parent={DATASETS_PARENT} />;
