// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useModelSearch,
  type UseModelSearchOptions,
  type UseModelSearchResult,
} from '@nemo/common/src/api/models/useModelSearch';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  defaultWizardValues,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import { WorkspaceSourceFields } from '@studio/routes/NewDeploymentRoute/WorkspaceSourceFields';
import { renderRoute } from '@studio/tests/util/render';
import type { FC } from 'react';
import { useForm } from 'react-hook-form';

vi.mock('@nemo/common/src/api/models/useModelSearch', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@nemo/common/src/api/models/useModelSearch')>();
  return { ...actual, useModelSearch: vi.fn() };
});

const mockUseModelSearch = vi.mocked(useModelSearch);

const emptyResult: UseModelSearchResult = {
  models: [],
  groups: [],
  search: '',
  error: null,
  loading: false,
  onSearchChange: vi.fn(),
  onLoadMore: vi.fn(),
  hasMore: false,
  isLoadingMore: false,
};

const buildModel = (overrides: Partial<ModelEntity> = {}) =>
  ({ id: 'model-1', name: 'my-model', workspace: 'default', ...overrides }) as ModelEntity;

/** Renders the workspace source fields against a real form control. */
const Harness: FC = () => {
  const { control, formState } = useForm<WizardFormValues>({
    defaultValues: defaultWizardValues(),
  });
  return (
    <WorkspaceSourceFields
      workspace="default"
      queryEnabled
      control={control}
      errors={formState.errors}
      onPickerTypeChange={() => {}}
    />
  );
};

const capturedOptions = (): UseModelSearchOptions => {
  const call = mockUseModelSearch.mock.calls.at(-1);
  if (!call) throw new Error('useModelSearch was never called');
  return call[0];
};

describe('WorkspaceSourceFields model picker', () => {
  beforeEach(() => {
    mockUseModelSearch.mockReturnValue(emptyResult);
  });

  it('passes an include predicate to the model search', () => {
    renderRoute(<Harness />);
    expect(capturedOptions().include).toBeDefined();
  });

  it('admits models that have a fileset of weights', () => {
    renderRoute(<Harness />);
    const include = capturedOptions().include!;
    expect(include(buildModel({ fileset: 'default/qwen3-0.6b-weights' }))).toBe(true);
  });

  it('excludes remote provider models, which have no weights to pull', () => {
    renderRoute(<Harness />);
    const include = capturedOptions().include!;
    // Shape of a NVIDIA Build catalog entry: served remotely, no fileset.
    expect(include(buildModel({ model_providers: ['default/build'] }))).toBe(false);
  });
});
