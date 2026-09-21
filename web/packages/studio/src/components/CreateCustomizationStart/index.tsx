// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner } from '@nvidia/foundations-react-core';
import {
  START_OPTIONS,
  TEMPLATE_GROUP_TITLE,
} from '@studio/components/CreateCustomizationStart/constants';
import { JsonConfigStep } from '@studio/components/CreateCustomizationStart/JsonConfigStep';
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
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // The JSON option is the only one that asks for anything else before the form.
  const [loadingJson, setLoadingJson] = useState(false);

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
    CUSTOMIZATION_TEMPLATES.find((template) => template.id === selectedId) ?? null;

  const handleContinue = async () => {
    if (selectedId === 'scratch') {
      onContinue({ optionId: 'scratch' });
      return;
    }
    if (selectedId === 'json') {
      setLoadingJson(true);
      return;
    }
    if (!selectedTemplate) return;

    // Registering the model and loading the dataset has to finish before the form can
    // reference them, so it happens here rather than on the next screen.
    const initialValues = await runTemplateSetup(selectedTemplate);
    // Provisioning spans a render, and the group is locked throughout, but only hand over
    // values that still match what is selected.
    if (initialValues && selectedId === selectedTemplate.id) {
      onContinue({ optionId: 'template', initialValues });
    }
  };

  if (loadingJson) {
    return (
      <JsonConfigStep
        onBack={() => setLoadingJson(false)}
        onContinue={(initialValues) => onContinue({ optionId: 'json', initialValues })}
      />
    );
  }

  return (
    <StartPage
      heading="Fine-tune a Model"
      headingDescription="Train a model on your own data. Start from a config you already have, pick a ready-made recipe, or set everything up yourself."
      options={START_OPTIONS}
      templateGroups={templateGroups}
      value={selectedId}
      // Provisioning registers models and uploads a dataset, which takes long enough that
      // the tiles would stay clickable behind the disabled Continue button. Moving the
      // selection then would leave a finished setup pointing at something else.
      onChange={setSelectedId}
      disabled={isSettingUp}
      canContinue={selectedId !== null && !isSettingUp}
      continueLabel={isSettingUp ? statusLabel : 'Continue'}
      continueLoading={isSettingUp}
      onContinue={() => void handleContinue()}
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
