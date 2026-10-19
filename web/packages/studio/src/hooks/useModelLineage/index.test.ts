// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { useCustomizationJobForModel } from '@studio/hooks/useCustomizationJobForModel';
import { useModelDeploymentStatuses } from '@studio/hooks/useModelDeploymentStatuses';
import { parseFilesetRef, useModelLineage } from '@studio/hooks/useModelLineage';
import { renderHook } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/platform/models');
vi.mock('@studio/hooks/useCustomizationJobForModel');
vi.mock('@studio/hooks/useModelDeploymentStatuses');

const mockGetModel = vi.mocked(useModelsGetModel);
const mockJob = vi.mocked(useCustomizationJobForModel);
const mockStatuses = vi.mocked(useModelDeploymentStatuses);

const adapter = {
  name: 'my-adapter',
  workspace: 'adapter-ws',
  fileset: 'my-adapter-files',
  finetuning_type: 'lora',
} as Adapter;

const model = {
  name: 'my-model',
  workspace: 'ws',
  fileset: 'files-ws/my-model-files',
  model_providers: ['ws/provider'],
  adapters: [adapter],
} as ModelEntity;

const mockModelQuery = (state: { data?: ModelEntity; isLoading?: boolean; isError?: boolean }) =>
  mockGetModel.mockReturnValue({
    isLoading: false,
    isError: false,
    ...state,
  } as unknown as ReturnType<typeof useModelsGetModel>);

describe('parseFilesetRef', () => {
  it('splits a workspace/name reference', () => {
    expect(parseFilesetRef('other-ws/weights', 'ws')).toEqual({
      workspace: 'other-ws',
      name: 'weights',
    });
  });

  it('places a bare name in the fallback workspace', () => {
    expect(parseFilesetRef('weights', 'ws')).toEqual({ workspace: 'ws', name: 'weights' });
  });

  it.each([undefined, '', 'ws/', '/weights'])('returns null for %j', (filesetRef) => {
    expect(parseFilesetRef(filesetRef, 'ws')).toBeNull();
  });
});

describe('useModelLineage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockJob.mockReturnValue({ jobName: 'job-1', isLoading: false });
    mockStatuses.mockImplementation(
      (targets) =>
        new Map(
          targets.map((target) => [
            target.key,
            { kind: 'served', hasDeployment: true, deploymentRef: 'ws/deployment' },
          ])
        )
    );
  });

  it("resolves a model's job, fileset and deployment", () => {
    mockModelQuery({ data: model });
    const { result } = renderHook(() => useModelLineage('ws', 'my-model'));

    expect(mockJob).toHaveBeenCalledWith(model);
    expect(result.current.job).toEqual({ jobName: 'job-1', workspace: 'ws', isLoading: false });
    expect(result.current.fileset).toEqual({ workspace: 'files-ws', name: 'my-model-files' });
    expect(result.current.deployment).toMatchObject({ deploymentRef: 'ws/deployment' });
  });

  it("resolves an adapter's links from the adapter, in the adapter's workspace", () => {
    mockModelQuery({ data: model });
    const { result } = renderHook(() => useModelLineage('ws', 'my-model', 'my-adapter'));

    expect(result.current.adapter).toBe(adapter);
    expect(mockJob).toHaveBeenCalledWith(adapter);
    expect(mockStatuses).toHaveBeenCalledWith([expect.objectContaining({ model, adapter })]);
    expect(result.current.job.workspace).toBe('adapter-ws');
    expect(result.current.fileset).toEqual({ workspace: 'adapter-ws', name: 'my-adapter-files' });
  });

  it('reports a missing adapter without falling back to the model', () => {
    mockModelQuery({ data: model });
    const { result } = renderHook(() => useModelLineage('ws', 'my-model', 'missing'));

    expect(result.current.isAdapterMissing).toBe(true);
    expect(mockJob).toHaveBeenCalledWith(null);
    expect(result.current.fileset).toBeNull();
    expect(result.current.deployment).toEqual({ kind: 'unknown' });
  });

  it('handles a model with no fileset or providers', () => {
    mockModelQuery({ data: { ...model, fileset: undefined, model_providers: [] } });
    mockJob.mockReturnValue({ jobName: undefined, isLoading: false });
    const { result } = renderHook(() => useModelLineage('ws', 'my-model'));

    expect(result.current.job.jobName).toBeUndefined();
    expect(result.current.fileset).toBeNull();
  });

  it('reports deployment as loading while the model loads, and unknown if it fails', () => {
    mockModelQuery({ isLoading: true });
    const { result, rerender } = renderHook(() => useModelLineage('ws', 'my-model'));
    expect(result.current.deployment).toEqual({ kind: 'loading' });

    mockModelQuery({ isError: true });
    rerender();
    expect(result.current.isError).toBe(true);
    expect(result.current.deployment).toEqual({ kind: 'unknown' });
  });
});
