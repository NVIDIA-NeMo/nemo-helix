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

const aliasesOf = (builder: Builder | null) =>
  builder?.getBuilderValues().models.map((model) => model.alias);

const snapshotOf = (builder: Builder | null) =>
  builder?.getBuilderValues().models.map(({ id, alias, model }) => ({ id, alias, model }));

describe('ModelConfigPanel', () => {
  it('leaves every model untouched when the open model changes', () => {
    const ref = mountPanel('model-0');
    const before = snapshotOf(ref.builder);

    act(() => ref.open('model-1'));
    act(() => ref.open('model-2'));
    act(() => ref.open('model-0'));

    expect(snapshotOf(ref.builder)).toEqual(before);
  });

  it('leaves every model with an alias when an earlier model is removed', () => {
    const ref = mountPanel('model-2');

    act(() => ref.builder?.removeModel('model-0'));

    expect(aliasesOf(ref.builder)).toEqual(['b', 'c']);
    expect(validateModels(ref.builder?.getBuilderValues().models ?? [])).toEqual([]);
  });

  it('leaves every model with an alias when the open model is removed', () => {
    const ref = mountPanel('model-1');

    act(() => ref.builder?.removeModel('model-1'));

    expect(aliasesOf(ref.builder)).toEqual(['a', 'c']);
    expect(validateModels(ref.builder?.getBuilderValues().models ?? [])).toEqual([]);
  });

  it('leaves every model with an alias when the last model is removed', () => {
    const ref = mountPanel('model-0');

    act(() => ref.builder?.removeModel('model-2'));
    act(() => ref.builder?.removeModel('model-1'));

    expect(aliasesOf(ref.builder)).toEqual(['a']);
  });
});
