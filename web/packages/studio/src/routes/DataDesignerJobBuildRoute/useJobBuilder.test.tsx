// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { modelsListModels } from '@nemo/sdk/generated/platform/models';
import type { ModelEntity, ModelEntitysPage } from '@nemo/sdk/generated/platform/schema';
import type { FilesetTemplate } from '@studio/components/CreateFilesetStart/types';
import { useJobBuilder } from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';
import { act, renderHook, waitFor } from '@testing-library/react';
import { StrictMode } from 'react';

vi.mock('@nemo/sdk/generated/platform/models', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nemo/sdk/generated/platform/models')>();
  return { ...actual, modelsListModels: vi.fn() };
});

const mockListModels = vi.mocked(modelsListModels);

const NEMOTRON = 'nvidia-llama-3-3-nemotron-super-49b-v1';

const makePage = (data: ModelEntity[]): ModelEntitysPage =>
  ({ data, pagination: { page: 1, total_pages: 1 } }) as ModelEntitysPage;

const model = (name: string, providers: string[] = ['ws1/build']): ModelEntity =>
  ({ id: name, name, workspace: 'ws1', model_providers: providers }) as unknown as ModelEntity;

/** A template that seeds a bare model name, the way the real templates do. */
const template = {
  id: 'text-to-python',
  models: [{ alias: 'default', model: NEMOTRON }],
  columns: [],
} as unknown as FilesetTemplate;

beforeEach(() => {
  mockListModels.mockReset();
  mockListModels.mockResolvedValue(makePage([model(NEMOTRON), model('some-other-model')]));
});

describe('useJobBuilder template auto-fill', () => {
  it('applies under StrictMode, where effects mount twice', async () => {
    const { result } = renderHook(() => useJobBuilder(template, 'ws1'), { wrapper: StrictMode });

    await waitFor(() =>
      expect(result.current.getBuilderValues().models[0]).toMatchObject({
        model: `ws1/${NEMOTRON}`,
        provider: 'ws1/build',
      })
    );
  });

  it('resolves each seeded model exactly once', async () => {
    const { result } = renderHook(() => useJobBuilder(template, 'ws1'), { wrapper: StrictMode });

    await waitFor(() =>
      expect(result.current.getBuilderValues().models[0].provider).toBe('ws1/build')
    );

    // One preferred-name lookup plus one first-page fallback, for the single seeded model.
    expect(mockListModels).toHaveBeenCalledTimes(2);
  });

  it('reports no issue when the template model resolves', async () => {
    const { result } = renderHook(() => useJobBuilder(template, 'ws1'), { wrapper: StrictMode });

    await waitFor(() =>
      expect(result.current.getBuilderValues().models[0].provider).toBe('ws1/build')
    );
    expect(result.current.templateModelIssues).toEqual([]);
  });

  it('keeps the requested model and reports it when the workspace serves a different one', async () => {
    mockListModels.mockResolvedValue(makePage([model('some-other-model')]));

    const { result } = renderHook(() => useJobBuilder(template, 'ws1'), { wrapper: StrictMode });

    await waitFor(() =>
      expect(result.current.templateModelIssues).toEqual([
        { id: 'model-0', alias: 'default', requested: NEMOTRON },
      ])
    );
    expect(result.current.getBuilderValues().models[0]).toMatchObject({
      model: NEMOTRON,
      provider: '',
    });
  });

  it('keeps the requested model when the workspace has nothing to resolve to', async () => {
    mockListModels.mockResolvedValue(makePage([]));

    const { result } = renderHook(() => useJobBuilder(template, 'ws1'), { wrapper: StrictMode });

    await waitFor(() =>
      expect(result.current.templateModelIssues).toEqual([
        { id: 'model-0', alias: 'default', requested: NEMOTRON },
      ])
    );
    expect(result.current.getBuilderValues().models[0].model).toBe(NEMOTRON);
  });

  it('reports a model whose only match cannot serve it', async () => {
    mockListModels.mockResolvedValue(makePage([model(NEMOTRON, [])]));

    const { result } = renderHook(() => useJobBuilder(template, 'ws1'), { wrapper: StrictMode });

    await waitFor(() =>
      expect(result.current.templateModelIssues).toEqual([
        { id: 'model-0', alias: 'default', requested: NEMOTRON },
      ])
    );
    expect(result.current.getBuilderValues().models[0].model).toBe(NEMOTRON);
  });

  it('surfaces a retryable error instead of a missing-model issue when the lookup rejects', async () => {
    mockListModels.mockRejectedValue(new Error('network error'));

    const { result } = renderHook(() => useJobBuilder(template, 'ws1'), { wrapper: StrictMode });

    await waitFor(() => expect(result.current.autoFillError).toBe(true));
    expect(result.current.templateModelIssues).toEqual([]);
    expect(result.current.getBuilderValues().models[0]).toMatchObject({
      model: NEMOTRON,
      provider: '',
    });

    mockListModels.mockResolvedValue(makePage([model(NEMOTRON)]));
    act(() => result.current.retryAutoFill());

    await waitFor(() => expect(result.current.autoFillError).toBe(false));
    expect(result.current.getBuilderValues().models[0]).toMatchObject({
      model: `ws1/${NEMOTRON}`,
      provider: 'ws1/build',
    });
    expect(result.current.templateModelIssues).toEqual([]);
  });

  it('leaves models that already carry a provider alone', async () => {
    const seeded = {
      ...template,
      models: [{ alias: 'default', model: NEMOTRON }],
    } as unknown as FilesetTemplate;

    const { result } = renderHook(
      () => useJobBuilder(seeded, 'ws1', { name: 'clone', rows: '10', columns: [], models: [] }),
      { wrapper: StrictMode }
    );

    await waitFor(() => expect(result.current.getBuilderValues().models).toEqual([]));
    expect(mockListModels).not.toHaveBeenCalled();
  });
});
