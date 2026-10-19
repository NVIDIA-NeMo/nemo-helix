// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toInferenceModelName } from '@nemo/common/src/utils/models';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Empty } from '@studio/components/Empty';
import { Loading } from '@studio/components/Layouts/Loading';
import { ModelChat } from '@studio/components/ModelChat';
import { DeployModelCta } from '@studio/components/ModelChat/DeployModelCta';
import { useModelChatAvailability } from '@studio/hooks/useModelChatAvailability';
import type { FC } from 'react';

export interface ModelChatTabProps {
  /** The model to chat with; for an adapter, the model it belongs to. */
  model: ModelEntity;
  adapter?: Adapter | null;
}

/** Chat with a model or adapter, or explain what has to be deployed before you can. */
export const ModelChatTab: FC<ModelChatTabProps> = ({ model, adapter }) => {
  const { modelChatStatus, isLoading, isAdapterUnserved } = useModelChatAvailability(model, {
    adapter,
  });
  const workspace = model.workspace;

  if (isLoading) {
    return <Loading />;
  }

  if (isAdapterUnserved) {
    return (
      <Empty
        title="Adapter is not currently served"
        description="No provider lists this adapter among its served models, so it cannot be chatted with yet. It becomes available once the deployment serving its base model has loaded the adapter."
        slotAction={
          <DeployModelCta modelRef={`${workspace}/${model.name}`} label="Deploy base model" />
        }
      />
    );
  }

  return (
    <ModelChat
      model={toInferenceModelName(model, adapter)}
      workspace={workspace}
      promptData={model.prompt}
      modelChatStatus={modelChatStatus}
      deployModelRef={`${workspace}/${model.name}`}
      deployModelLabel={adapter ? 'Deploy base model' : undefined}
    />
  );
};
