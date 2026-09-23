// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner } from '@nvidia/foundations-react-core';
import {
  START_OPTIONS,
  TEMPLATES_TAG,
  TEMPLATE_GROUP_TITLE,
} from '@studio/components/CreateCustomizationStart/constants';
import type { CreateCustomizationStartProps } from '@studio/components/CreateCustomizationStart/types';
import { useTemplateSetup } from '@studio/components/CreateCustomizationStart/useTemplateSetup';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';
import { CUSTOMIZATION_TEMPLATES } from '@studio/constants/customizationTemplates';
import { Box } from 'lucide-react';
import { useMemo, useState, type FC } from 'react';

export const CreateCustomizationStart: FC<CreateCustomizationStartProps> = ({
  workspace,
  onContinue,
}) => {
  // Only to mark the tile that is provisioning; nothing stays selected after a pick.
  const [busyId, setBusyId] = useState<string | null>(null);

  const { run: runTemplateSetup, statusLabel, error: templateError } = useTemplateSetup(workspace);
  const isSettingUp = statusLabel !== '';

  const templateGroups = useMemo<StartTemplateGroup[]>(
    () => [
      {
        id: 'nvidia-recipes',
        title: TEMPLATE_GROUP_TITLE,
        templates: CUSTOMIZATION_TEMPLATES.map((template) => ({
          id: template.id,
          name: template.title,
          description: template.description,
          icon: Box,
        })),
      },
    ],
    []
  );

  const handleSelect = async (id: string) => {
    if (id === 'scratch') {
      onContinue({ optionId: 'scratch' });
      return;
    }
    const template = CUSTOMIZATION_TEMPLATES.find((candidate) => candidate.id === id);
    if (!template) return;

    // Registering the model and loading the dataset has to finish before the form can
    // reference them, so it happens here rather than on the next screen.
    setBusyId(id);
    try {
      const initialValues = await runTemplateSetup(template);
      if (initialValues) {
        onContinue({ optionId: 'template', initialValues });
      }
    } finally {
      setBusyId(null);
    }
  };

  return (
    <StartPage
      heading="Fine-tune a Model"
      headingDescription="Train a model on your own data. Pick a ready-made recipe, or set everything up yourself."
      options={START_OPTIONS}
      templateGroups={templateGroups}
      templatesTag={TEMPLATES_TAG}
      onSelect={(id) => void handleSelect(id)}
      // Provisioning registers models and uploads a dataset, which takes long enough that
      // the other tiles would stay clickable and start a second setup over the first.
      disabled={isSettingUp}
      busyId={busyId}
      busyLabel={isSettingUp ? statusLabel : undefined}
      slotBanner={
        templateError ? (
          <Banner kind="inline" status="error">
            {templateError}
          </Banner>
        ) : null
      }
    />
  );
};
