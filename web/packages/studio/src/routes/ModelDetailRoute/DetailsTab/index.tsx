// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isBaseModel } from '@nemo/common/src/utils/models';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Grid, Stack } from '@nvidia/foundations-react-core';
import type { ModelLineage } from '@studio/hooks/useModelLineage';
import { ModelDetailTab } from '@studio/routes/ModelDetailRoute/constants';
import { AdaptersSection } from '@studio/routes/ModelDetailRoute/DetailsTab/AdaptersSection';
import {
  getAdapterFields,
  getAdapterTiles,
  getArchitectureFields,
  getArchitectureTiles,
  getModelFields,
  getModelTiles,
} from '@studio/routes/ModelDetailRoute/DetailsTab/fields';
import { InferenceSection } from '@studio/routes/ModelDetailRoute/DetailsTab/InferenceSection';
import { DetailPanel } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/DetailPanel';
import { LineageSection } from '@studio/routes/ModelDetailRoute/DetailsTab/LineageSection';
import { PromptSection } from '@studio/routes/ModelDetailRoute/DetailsTab/PromptSection';
import { getModelDetailRoute } from '@studio/routes/utils';
import type { FC } from 'react';

export interface DetailsTabProps {
  workspace: string;
  model: ModelEntity;
  adapter: Adapter | null;
  lineage: ModelLineage;
}

export const DetailsTab: FC<DetailsTabProps> = ({ workspace, model, adapter, lineage }) => {
  const adapters = model.adapters ?? [];
  // A base model has no base or training job of its own; an adapter page describes the adapter.
  const isBase = !adapter && isBaseModel(model);
  const baseCard = isBase
    ? undefined
    : adapter
      ? {
          name: model.name,
          to: getModelDetailRoute(workspace, model.name),
          note: 'The model this adapter applies to',
        }
      : { name: model.base_model, note: 'Checkpoint it was fine-tuned from' };
  const jobCard = isBase
    ? undefined
    : {
        ...lineage.job,
        to: getModelDetailRoute(workspace, model.name, {
          adapterName: adapter?.name,
          tab: ModelDetailTab.Training,
        }),
      };

  return (
    <Stack gap="density-3xl" className="w-full pb-density-2xl">
      <LineageSection
        base={baseCard}
        job={jobCard}
        fileset={lineage.fileset}
        deployment={lineage.deployment}
        isAdapter={Boolean(adapter)}
      />

      <Grid cols={{ base: 1, lg: 2 }} gap="density-xl">
        {adapter ? (
          <>
            <DetailPanel
              heading="Adapter"
              tiles={getAdapterTiles(adapter)}
              fields={getAdapterFields(adapter)}
            />
            <DetailPanel
              heading="Base Model"
              description={`${model.name}, the model this adapter is applied to.`}
              tiles={getModelTiles(model)}
              fields={getModelFields(model)}
            />
          </>
        ) : (
          <>
            <DetailPanel
              heading="Model"
              tiles={getModelTiles(model)}
              fields={getModelFields(model)}
            />
            {model.spec && (
              <DetailPanel
                heading="Architecture"
                tiles={getArchitectureTiles(model.spec)}
                fields={getArchitectureFields(model.spec)}
              />
            )}
          </>
        )}
      </Grid>

      {!adapter && adapters.length > 0 && (
        <AdaptersSection workspace={workspace} modelName={model.name} adapters={adapters} />
      )}

      {model.api_endpoint && <InferenceSection model={model} />}
      {(model.prompt?.system_prompt || model.prompt?.icl_few_shot_examples) && (
        <PromptSection model={model} />
      )}
      {adapter && model.spec && (
        <DetailPanel
          heading="Architecture"
          description={`The network shape of ${model.name}.`}
          tiles={getArchitectureTiles(model.spec)}
          fields={getArchitectureFields(model.spec)}
        />
      )}
    </Stack>
  );
};
