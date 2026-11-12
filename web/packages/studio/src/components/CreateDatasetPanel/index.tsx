// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';

export interface CreateDatasetPanelProps {
  open: boolean;
  workspace: string;
  onClose: () => void;
  onCreated: (fileset: FilesetOutput) => void;
}

// eslint-disable-next-line unused-imports/no-unused-vars
export function CreateDatasetPanel(_props: CreateDatasetPanelProps) {
  return null;
}
