// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import { useResolvedAutomodelRecipe } from '@studio/components/NewCustomizationForm/useResolvedAutomodelRecipe';
import { renderHook } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/platform/models', () => ({ useModelsGetModel: vi.fn() }));

const mockGetModel = vi.mocked(useModelsGetModel);

const returnsModel = (spec: unknown) =>
  mockGetModel.mockReturnValue({ data: spec ? { spec } : undefined } as ReturnType<
    typeof useModelsGetModel
  >);

const queryOptions = () => mockGetModel.mock.lastCall?.[3]?.query;

describe('useResolvedAutomodelRecipe', () => {
  beforeEach(() => {
    mockGetModel.mockReset();
  });

  it('resolves auto from the base model it fetched', () => {
    returnsModel({ head_type: 'cross_encoder' });
    const { result } = renderHook(() =>
      useResolvedAutomodelRecipe('default/reranker', 'auto', true)
    );
    expect(result.current).toBe('cross_encoder');
    expect(mockGetModel.mock.lastCall?.slice(0, 2)).toEqual(['default', 'reranker']);
    expect(queryOptions()?.enabled).toBe(true);
  });

  it('is unresolved until the model loads, or when it fails to load', () => {
    returnsModel(undefined);
    const { result } = renderHook(() =>
      useResolvedAutomodelRecipe('default/embedder', 'auto', true)
    );
    expect(result.current).toBeNull();
  });

  it('is unresolved on auto before a model is chosen', () => {
    returnsModel(undefined);
    const { result } = renderHook(() => useResolvedAutomodelRecipe('', 'auto', true));
    expect(result.current).toBeNull();
  });

  it('treats an omitted recipe as auto', () => {
    returnsModel({ head_type: 'embedding' });
    const { result } = renderHook(() =>
      useResolvedAutomodelRecipe('default/embedder', undefined, true)
    );
    expect(result.current).toBe('bi_encoder');
    expect(queryOptions()?.enabled).toBe(true);
  });

  it('stays auto for a loaded causal LM', () => {
    returnsModel({ head_type: 'causal_lm' });
    const { result } = renderHook(() => useResolvedAutomodelRecipe('default/llama', 'auto', true));
    expect(result.current).toBe('auto');
  });

  it('passes an explicit recipe or another backend through untouched', () => {
    returnsModel(undefined);
    expect(
      renderHook(() => useResolvedAutomodelRecipe('default/embedder', 'sft', true)).result.current
    ).toBe('sft');
    expect(
      renderHook(() => useResolvedAutomodelRecipe('default/embedder', 'auto', false)).result.current
    ).toBe('auto');
  });

  it('does not fetch for an explicit recipe, another backend, or no model', () => {
    returnsModel(undefined);
    renderHook(() => useResolvedAutomodelRecipe('default/embedder', 'sft', true));
    expect(queryOptions()?.enabled).toBe(false);
    renderHook(() => useResolvedAutomodelRecipe('default/embedder', 'auto', false));
    expect(queryOptions()?.enabled).toBe(false);
    renderHook(() => useResolvedAutomodelRecipe('', 'auto', true));
    expect(queryOptions()?.enabled).toBe(false);
  });
});
