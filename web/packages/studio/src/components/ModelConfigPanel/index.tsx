// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ModelConfigFields,
  type ModelConfigFieldsProps,
} from '@studio/components/ModelConfigPanel/ModelConfigFields';
import type { JobBuilderFormValues } from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';
import type { FC } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export interface ModelConfigPanelProps extends Omit<ModelConfigFieldsProps, 'modelIndex'> {
  modelId: string;
}

export const ModelConfigPanel: FC<ModelConfigPanelProps> = ({ modelId, ...props }) => {
  const { control } = useFormContext<JobBuilderFormValues>();
  const modelIds = useWatch({
    control,
    name: 'models',
    compute: (models) => models.map((model) => model.id),
  });
  const modelIndex = modelIds.indexOf(modelId);
  return modelIndex < 0 ? null : (
    <ModelConfigFields key={modelId} {...props} modelIndex={modelIndex} />
  );
};
