// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  activeRolesForStrategy,
  DETECTOR_ROLE,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import { useAnonymizerModels } from '@studio/routes/AnonymizerBuilderRoute/useAnonymizerModels';
import { isNerDetectorModel } from '@studio/routes/AnonymizerBuilderRoute/utils';
import { pickDefaultModelName } from '@studio/util/buildSuggestedModelOptions';
import { useEffect, useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export const useDefaultRoleModels = (): { isLoading: boolean } => {
  const { control, setValue, getValues } = useFormContext<AnonymizerFormData>();
  const strategy = useWatch({ control, name: 'strategy' });
  const { models, isLoading, applyModel } = useAnonymizerModels();

  const roles = useMemo(() => activeRolesForStrategy(strategy), [strategy]);

  useEffect(() => {
    if (!models.length) return;
    const chatModels = models.filter((model) => !isNerDetectorModel(model));
    const suggestedName = pickDefaultModelName(
      chatModels.map((model) => ({ name: model.served_model_name ?? model.name }))
    );
    const llm =
      chatModels.find((model) => (model.served_model_name ?? model.name) === suggestedName) ??
      chatModels[0];
    const detector = models.find(isNerDetectorModel);
    for (const role of roles) {
      const current = getValues(`roleModels.${role}.modelId`);
      if (current) continue;
      const pick = role === DETECTOR_ROLE ? detector : llm;
      if (!pick) continue;
      setValue(`roleModels.${role}.modelId`, pick.id);
      applyModel(role, pick.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models, roles, getValues, setValue]);

  return { isLoading };
};
