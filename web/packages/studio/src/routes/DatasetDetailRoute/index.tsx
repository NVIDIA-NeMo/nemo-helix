// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import { FilesetDetailRoute } from '@studio/routes/FilesetDetailRoute';
import type { FilesetDetailParent } from '@studio/routes/FilesetDetailRoute/FilesetDetailParent';
import { getDatasetDetailRoute, getWorkspaceDatasetsRoute } from '@studio/routes/utils';
import type { FC } from 'react';

const DATASETS_PARENT: FilesetDetailParent = {
  label: 'Datasets',
  purpose: FilesetPurpose.dataset,
  getListRoute: getWorkspaceDatasetsRoute,
  getDetailRoute: getDatasetDetailRoute,
};

/** The fileset detail page for datasets, with breadcrumbs and links kept under /datasets. */
export const DatasetDetailRoute: FC = () => <FilesetDetailRoute parent={DATASETS_PARENT} />;
