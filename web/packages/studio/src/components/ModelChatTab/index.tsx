// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toInferenceModelName } from '@nemo/common/src/utils/models';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Loading } from '@studio/components/Layouts/Loading';
import { ModelChat } from '@studio/components/ModelChat';
import { ChatUnavailable } from '@studio/components/ModelChat/ChatUnavailable';
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
      <ChatUnavailable
        status={modelChatStatus}
        isAdapterUnserved
        deployModelRef={`${workspace}/${model.name}`}
      />
    );
  }

  return (
    <ModelChat
      // An adapter goes on the wire as `base&adapters/{ws}/{name}`, which the
      // gateway resolves through the base model's VirtualModel — so it needs no
      // special base URL and inherits that VM's middleware.
      model={toInferenceModelName(model, adapter)}
      workspace={workspace}
      promptData={model.prompt}
      modelChatStatus={modelChatStatus}
      // `model` is the base model in both branches. Deploying it is what makes
      // an adapter servable, so the adapter case links to the same wizard —
      // only the copy changes, since "this model" would name the wrong entity.
      deployModelRef={`${workspace}/${model.name}`}
      deployModelLabel={adapter ? 'Deploy base model' : undefined}
    />
  );
};
