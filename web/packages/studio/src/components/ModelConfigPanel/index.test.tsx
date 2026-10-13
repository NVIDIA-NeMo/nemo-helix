// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelConfigPanel } from '@studio/components/ModelConfigPanel';
import { validateModels } from '@studio/routes/DataDesignerJobBuildRoute/models';
import {
  type JobBuilderSeed,
  useJobBuilder,
} from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';
import { act, render } from '@testing-library/react';
import { type FC, useState } from 'react';
import { FormProvider } from 'react-hook-form';

vi.mock('@nemo/common/src/components/ModelSelectV2/WorkspaceModelSelect', () => ({
  WorkspaceModelSelect: () => null,
}));

const seed: JobBuilderSeed = {
  name: 'ds',
  rows: '10',
  columns: [],
  models: ['a', 'b', 'c'].map((alias, index) => ({
    id: `model-${index}`,
    alias,
    model: `ws/${alias}`,
    provider: 'ws/build',
    inferenceParams: {},
  })),
};

type Builder = ReturnType<typeof useJobBuilder>;

const mountPanel = (openModelId: string) => {
  const ref: { builder: Builder | null; open: (modelId: string) => void } = {
    builder: null,
    open: () => undefined,
  };
  const Harness: FC = () => {
    const [modelId, setModelId] = useState(openModelId);
    ref.open = setModelId;
    const builder = useJobBuilder(null, 'ws', seed);
    ref.builder = builder;
    return (
      <FormProvider {...builder.form}>
        <ModelConfigPanel
          modelId={modelId}
          workspace="ws"
          onRemove={() => undefined}
          onClose={() => undefined}
        />
      </FormProvider>
    );
  };
  render(<Harness />);
  return ref;
};

const modelsOf = (builder: Builder | null) => builder?.getBuilderValues().models ?? [];

const withoutIds = (builder: Builder | null, ...ids: string[]) =>
  modelsOf(builder).filter((model) => !ids.includes(model.id));

describe('ModelConfigPanel', () => {
  it('leaves every model untouched when the open model changes', () => {
    const ref = mountPanel('model-0');
    const before = modelsOf(ref.builder);

    act(() => ref.open('model-1'));
    act(() => ref.open('model-2'));
    act(() => ref.open('model-0'));

    expect(modelsOf(ref.builder)).toEqual(before);
  });

  it('leaves the remaining models untouched when a model before the open one is removed', () => {
    const ref = mountPanel('model-1');
    const expected = withoutIds(ref.builder, 'model-0');

    act(() => ref.builder?.removeModel('model-0'));

    expect(modelsOf(ref.builder)).toEqual(expected);
    expect(validateModels(modelsOf(ref.builder))).toEqual([]);
  });

  it('leaves the remaining models untouched when the open model is removed', () => {
    const ref = mountPanel('model-1');
    const expected = withoutIds(ref.builder, 'model-1');

    act(() => ref.builder?.removeModel('model-1'));

    expect(modelsOf(ref.builder)).toEqual(expected);
    expect(validateModels(modelsOf(ref.builder))).toEqual([]);
  });
});
