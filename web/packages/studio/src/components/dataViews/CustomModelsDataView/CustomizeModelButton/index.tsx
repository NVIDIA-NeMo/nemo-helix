// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { CreateButton } from '@studio/components/common/CreateButton';
import { useModelCustomizationEligibility } from '@studio/hooks/useModelCustomizationEligibility';
import { getNewCustomizationFormRoute, getNewCustomizationJobRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

export interface CustomizeModelButtonProps {
  workspace: string;
  /**
   * When provided, the button is shown in the per-model context: label becomes
   * "Customize this Model", a loading spinner is shown while eligibility is
   * being checked, and the button is disabled if the model has no fileset to
   * fine-tune from.
   */
  model?: ModelEntity;
}

export const CustomizeModelButton: FC<CustomizeModelButtonProps> = ({ workspace, model }) => {
  const navigate = useNavigate();
  const { canFineTune, isLoading } = useModelCustomizationEligibility(model);

  const goToFineTuning = () =>
    navigate(
      // Naming a model already answers "how do you want to start?" — skip the picker.
      model
        ? getNewCustomizationFormRoute(workspace, { model: getURNFromNamedEntityRef(model) })
        : getNewCustomizationJobRoute(workspace)
    );

  return model ? (
    <LoadingButton
      kind="primary"
      size="small"
      className="flex-1"
      height={28}
      onClick={goToFineTuning}
      loading={isLoading}
      disabled={!canFineTune}
    >
      Customize this Model
    </LoadingButton>
  ) : (
    <CreateButton onClick={goToFineTuning}>Customize a Model</CreateButton>
  );
};
