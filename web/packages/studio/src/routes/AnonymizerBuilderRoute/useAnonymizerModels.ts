// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsListProviders } from '@nemo/sdk/generated/platform/model-providers';
import {
  modelsFromProviders,
  type DataDesignerModelOption,
} from '@studio/components/NewDataDesignerJobForm/utils';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import { useCallback, useMemo } from 'react';
import { useFormContext } from 'react-hook-form';

interface AnonymizerModels {
  readonly models: DataDesignerModelOption[];
  readonly items: { label: string; value: string }[];
  readonly isLoading: boolean;
  readonly applyModel: (role: string, id: string) => void;
}

/** Workspace models plus the setter that writes a role's pick into the form. */
export const useAnonymizerModels = (): AnonymizerModels => {
  const { setValue } = useFormContext<AnonymizerFormData>();
  const workspace = useWorkspaceFromPath();

  const { data: providersPage, isLoading } = useModelsListProviders(
    workspace,
    { page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: {} }
  );

  const models = useMemo(
    () => modelsFromProviders(providersPage?.data ?? []),
    [providersPage?.data]
  );
  const items = useMemo(
    () => models.map((model) => ({ label: model.name, value: model.id })),
    [models]
  );

  const applyModel = useCallback(
    (role: string, id: string) => {
      const selected = models.find((model) => model.id === id);
      setValue(`roleModels.${role}.model`, selected?.served_model_name ?? '', {
        shouldValidate: true,
      });
      setValue(`roleModels.${role}.provider`, selected?.model_providers?.[0] ?? '', {
        shouldValidate: true,
      });
    },
    [models, setValue]
  );

  return { models, items, isLoading, applyModel };
};
