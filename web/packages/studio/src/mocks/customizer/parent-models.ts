// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';

export const parentModel1: ModelEntity = {
  id: 'model-parent-1',
  name: 'codellama-70b-config',
  workspace: 'default',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

export const parentModel2: ModelEntity = {
  id: 'model-parent-2',
  name: 'gemma-7b-config',
  workspace: 'default',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

export const parentModel3: ModelEntity = {
  id: 'model-parent-3',
  name: 'gpt-43b-002-config',
  workspace: 'default',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

export const parentModels = [parentModel1, parentModel2, parentModel3];

export const getAvailableParentModelsResponse = {
  data: parentModels,
};
