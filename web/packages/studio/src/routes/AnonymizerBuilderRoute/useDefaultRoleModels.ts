// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  activeRolesForStrategy,
  BUILTIN_DETECTOR_VALUE,
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

  // The entity detector defaults to the platform's built-in in-process GLiNER2 (a UI-only sentinel
  // that makes the request omit entity_detector). This runs independently of the workspace model
  // list — the built-in needs none — so it also covers an empty workspace. applyModel fills in the
  // (empty) model/provider the role schema requires, matching the picker's own onChange path.
  useEffect(() => {
    if (roles.includes(DETECTOR_ROLE) && !getValues(`roleModels.${DETECTOR_ROLE}.modelId`)) {
      setValue(`roleModels.${DETECTOR_ROLE}.modelId`, BUILTIN_DETECTOR_VALUE);
      applyModel(DETECTOR_ROLE, BUILTIN_DETECTOR_VALUE);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roles, getValues, setValue]);

  useEffect(() => {
    if (!models.length) return;
    const chatModels = models.filter((model) => !isNerDetectorModel(model));
    const suggestedName = pickDefaultModelName(
      chatModels.map((model) => ({ name: model.served_model_name ?? model.name }))
    );
    const llm =
      chatModels.find((model) => (model.served_model_name ?? model.name) === suggestedName) ??
      chatModels[0];
    if (!llm) return;
    // The detector is handled above; every other role auto-picks a chat model.
    for (const role of roles) {
      if (role === DETECTOR_ROLE) continue;
      if (getValues(`roleModels.${role}.modelId`)) continue;
      setValue(`roleModels.${role}.modelId`, llm.id);
      applyModel(role, llm.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models, roles, getValues, setValue]);

  return { isLoading };
};
