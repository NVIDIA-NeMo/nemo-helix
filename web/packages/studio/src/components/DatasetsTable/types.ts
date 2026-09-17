// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import {
  type FilesetOutput as Dataset,
  type FilesetPurpose,
  type GithubStorageConfig,
  type HuggingfaceStorageConfig,
  type LocalStorageConfig,
  type NGCStorageConfig,
  type S3StorageConfig,
} from '@nemo/sdk/generated/platform/schema';
import { type ComponentProps, type ReactNode } from 'react';

export type ModalOpenState = 'delete' | 'edit' | 'none';

export type StorageConfig =
  | LocalStorageConfig
  | NGCStorageConfig
  | HuggingfaceStorageConfig
  | S3StorageConfig
  | GithubStorageConfig;

export type DatasetWithId = Dataset & { id: string };

export interface DatasetsTableProps {
  /** Callback when datasets are selected */
  onDatasetsSelected?: (datasets: Dataset[]) => void;
  /** Callback when a row is clicked */
  onRowClick?: (dataset: Dataset) => void;
  /** Disable row actions (default: true) */
  enableActions?: boolean;
  /** Enable bulk delete when items selected (default: false) */
  enableBulkDelete?: boolean;
  /** Enable search bar and filters (default: false) */
  enableFilters?: boolean;
  /** Enable checkbox selection (default: true) */
  enableSelection?: boolean;
  /** Type of selection (default: 'multiple') */
  selectionType?: 'multiple' | 'single';
  /** Render dataset name as link - provide a function that returns the route */
  getDatasetRoute?: (dataset: Dataset) => string;
  /** When set, restricts the fetched filesets to the given purpose. Pass FilesetPurpose.dataset in picker contexts that are specifically designed for dataset inputs. */
  purposeFilter?: FilesetPurpose;
  /** Custom render for row actions */
  renderRowActions?: (
    dataset: Dataset,
    callbacks: {
      onNavigate: () => void;
      onEdit: () => void;
      onDelete: () => void;
      onDatasetDeleted: (dataset: Dataset) => void;
    }
  ) => ReactNode;
  attributes?: {
    DataViewRoot?: ComponentProps<typeof DataView.Root<DatasetWithId>> & { dataMode: 'manual' };
    DataViewContent?: ComponentProps<typeof DataView.TableContent>;
  };
}
