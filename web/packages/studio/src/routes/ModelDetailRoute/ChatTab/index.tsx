// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Loading } from '@studio/components/Layouts/Loading';
import { ModelChatTab } from '@studio/components/ModelChatTab';
import { ComparePlayground } from '@studio/components/ModelCompareChat/ComparePlayground';
import { useCompareModels } from '@studio/hooks/useCompareModels';
import { useModelChatAvailability } from '@studio/hooks/useModelChatAvailability';
import type { FC } from 'react';

export interface ChatTabProps {
  model: ModelEntity;
  adapter: Adapter | null;
}

/**
 * The Playground's compare view, with this model locked in as the baseline. Until it can be
 * chatted with, falls back to the plain chat tab, which explains what to deploy.
 */
export const ChatTab: FC<ChatTabProps> = ({ model, adapter }) => {
  const { modelChatStatus, isLoading, isAdapterUnserved } = useModelChatAvailability(model, {
    adapter,
  });
  const { models, addModel, removeModel, setModel } = useCompareModels([
    {
      modelURN: `${model.workspace}/${model.name}`,
      adapter: adapter?.name ?? null,
      locked: true,
      promptData: model.prompt,
    },
  ]);

  if (isLoading) {
    return <Loading />;
  }

  if (isAdapterUnserved || modelChatStatus !== 'enabled') {
    return (
      <div className="mx-auto h-full w-full max-w-[768px]">
        <ModelChatTab model={model} adapter={adapter} />
      </div>
    );
  }

  return (
    <div className="h-full w-full">
      <ComparePlayground
        workspace={model.workspace}
        models={models}
        onRemoveModel={removeModel}
        onSetModel={setModel}
        onAddModel={addModel}
      />
    </div>
  );
};
