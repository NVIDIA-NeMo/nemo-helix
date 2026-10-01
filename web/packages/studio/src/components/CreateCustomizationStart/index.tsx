// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Stack } from '@nvidia/foundations-react-core';
import {
  START_OPTIONS,
  TEMPLATE_GROUP_TITLE,
} from '@studio/components/CreateCustomizationStart/constants';
import type {
  CreateCustomizationStartProps,
  StartOptionId,
} from '@studio/components/CreateCustomizationStart/types';
import { useTemplateSetup } from '@studio/components/CreateCustomizationStart/useTemplateSetup';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';
import { CUSTOMIZATION_TEMPLATES } from '@studio/constants/customizationTemplates';
import { Box } from 'lucide-react';
import { useMemo, useState, type FC } from 'react';

/** Templates are the middle rung, and the likeliest way in, so the page opens on them. */
const DEFAULT_OPTION: StartOptionId = 'template';

export const CreateCustomizationStart: FC<CreateCustomizationStartProps> = ({
  workspace,
  onContinue,
}) => {
  const [selectedId, setSelectedId] = useState<StartOptionId>(DEFAULT_OPTION);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);

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

  const selectedTemplate =
    CUSTOMIZATION_TEMPLATES.find((template) => template.id === selectedTemplateId) ?? null;

  const handleContinue = async () => {
    if (selectedId === 'scratch') {
      onContinue({ optionId: 'scratch' });
      return;
    }
    if (!selectedTemplate) return;

    // Registering the model and loading the dataset has to finish before the form can
    // reference them, so it happens here rather than on the next screen.
    const initialValues = await runTemplateSetup(selectedTemplate);
    // Provisioning spans a render, and the page is locked throughout, but only hand over
    // values that still match what is selected.
    if (initialValues && selectedTemplateId === selectedTemplate.id) {
      onContinue({ optionId: 'template', initialValues });
    }
  };

  return (
    <StartPage
      heading="Fine-tune a Model"
      headingDescription="Train a model on your own data. Pick a ready-made recipe, or set everything up yourself."
      options={START_OPTIONS}
      value={selectedId}
      onChange={(id) => {
        setSelectedId(id as StartOptionId);
        setSelectedTemplateId(null);
      }}
      // Provisioning registers models and uploads a dataset, which takes long enough that
      // the cards would stay clickable behind the disabled Continue. Moving the selection
      // then would leave a finished setup pointing at something else.
      disabled={isSettingUp}
      canContinue={!isSettingUp && (selectedId === 'scratch' || selectedTemplateId !== null)}
      continueLabel={isSettingUp ? statusLabel : 'Continue'}
      continueLoading={isSettingUp}
      onContinue={() => void handleContinue()}
      blockedHint={selectedId === 'template' ? 'Pick a recipe to continue.' : undefined}
      slotDetail={
        selectedId === 'template' ? (
          <Stack gap="density-2xl" className="w-full">
            <TemplateGroups
              groups={templateGroups}
              value={selectedTemplateId}
              onChange={setSelectedTemplateId}
              disabled={isSettingUp}
            />
          </Stack>
        ) : null
      }
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
