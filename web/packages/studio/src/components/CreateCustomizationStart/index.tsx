// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import {
  Banner,
  Block,
  Flex,
  Grid,
  GridItem,
  PageHeader,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  getCustomizationJobTemplatesQueryKey,
  listCustomizationJobTemplates,
  templateToFormFields,
} from '@studio/api/customization-job-templates/customizationJobTemplates';
import { START_OPTIONS } from '@studio/components/CreateCustomizationStart/constants';
import { StartOptionDetail } from '@studio/components/CreateCustomizationStart/StartOptionDetail';
import type {
  CreateCustomizationStartProps,
  StartOptionId,
} from '@studio/components/CreateCustomizationStart/types';
import { useTemplateSetup } from '@studio/components/CreateCustomizationStart/useTemplateSetup';
import { StartOptionCard } from '@studio/components/StartOptions/StartOptionCard';
import { CUSTOMIZATION_TEMPLATES } from '@studio/constants/customizationTemplates';
import { useQuery } from '@tanstack/react-query';
import { ArrowRight } from 'lucide-react';
import { useState, type FC } from 'react';

/** Why Continue is unavailable, shown next to the disabled button. */
const BLOCKED_HINT: Partial<Record<StartOptionId, string>> = {
  template: 'Pick a recipe to continue.',
  saved: 'Pick a saved template to continue.',
};

export const CreateCustomizationStart: FC<CreateCustomizationStartProps> = ({
  workspace,
  onContinue,
}) => {
  const [selectedId, setSelectedId] = useState<StartOptionId | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);

  const { run: runTemplateSetup, statusLabel, error: templateError } = useTemplateSetup(workspace);
  const isSettingUp = statusLabel !== '';

  // Same query key as the grid, so this reads the grid's cache rather than refetching.
  const { data: savedPage } = useQuery({
    queryKey: getCustomizationJobTemplatesQueryKey(workspace),
    queryFn: ({ signal }) => listCustomizationJobTemplates(workspace, undefined, signal),
    enabled: selectedId === 'saved',
  });

  const selectedOption = START_OPTIONS.find((option) => option.id === selectedId) ?? null;

  const selectedSavedTemplate =
    selectedId === 'saved' && selectedTemplateId
      ? savedPage?.data.find((template) => template.name === selectedTemplateId)
      : undefined;

  /**
   * Form values for the picked saved template, or `undefined` when none is picked or its
   * payload cannot be read — a template written by a newer Studio build, say. Continue stays
   * disabled in that case rather than opening a form seeded with nothing.
   */
  const selectedSavedFields = selectedSavedTemplate
    ? templateToFormFields(selectedSavedTemplate)
    : undefined;

  const selectOption = (optionId: StartOptionId) => {
    // Provisioning registers models and uploads a dataset, which takes long enough that the
    // cards stay clickable behind the disabled Continue button. Changing the selection then
    // would leave a finished setup pointing at a recipe the user has moved off.
    if (isSettingUp) return;
    setSelectedId(optionId);
    setSelectedTemplateId(null);
  };

  // A tile, plus that option's own payload.
  const canContinue =
    selectedOption !== null &&
    !isSettingUp &&
    (selectedOption.id !== 'template' || selectedTemplateId !== null) &&
    (selectedOption.id !== 'saved' || selectedSavedFields !== undefined);

  const handleContinue = async () => {
    if (!selectedOption) return;
    if (selectedOption.id === 'scratch') {
      onContinue({ optionId: 'scratch' });
      return;
    }
    if (selectedOption.id === 'saved') {
      // Saved templates only reference models and datasets that already exist, so unlike a
      // curated recipe there is nothing to provision — hand the fields straight over.
      if (selectedSavedFields)
        onContinue({ optionId: 'saved', initialValues: selectedSavedFields });
      return;
    }
    if (selectedOption.id !== 'template' || !selectedTemplateId) return;

    const template = CUSTOMIZATION_TEMPLATES.find((t) => t.id === selectedTemplateId);
    if (!template) return;
    // Registering the model and loading the dataset has to finish before the form can
    // reference them, so it happens here rather than on the next screen.
    const initialValues = await runTemplateSetup(template);
    // The guard above stops the selection moving, but the await still spans a render — only
    // hand over values that match what is selected now.
    if (initialValues && selectedTemplateId === template.id) {
      onContinue({ optionId: 'template', initialValues });
    }
  };

  return (
    <Stack className="h-full">
      <Block className="flex-1 overflow-auto">
        <Stack gap="density-2xl" padding="density-2xl">
          <PageHeader
            slotHeading="Fine-tune a Model"
            slotDescription="Train a model on your own data. Start from a config you already have, pick a ready-made recipe, or set everything up yourself."
          />

          <Stack gap="density-md">
            <Text kind="label/bold/sm" className="text-secondary">
              How do you want to start?
            </Text>
            <Grid colMinWidth="200px" gap="density-md">
              {START_OPTIONS.map((option) => (
                <GridItem key={option.id}>
                  <StartOptionCard
                    compact
                    option={option}
                    selected={selectedId === option.id}
                    onSelect={() => selectOption(option.id)}
                  />
                </GridItem>
              ))}
            </Grid>
          </Stack>

          {selectedOption ? (
            <StartOptionDetail
              option={selectedOption}
              workspace={workspace}
              selectedTemplateId={selectedTemplateId}
              onSelectTemplate={(id) => {
                if (!isSettingUp) setSelectedTemplateId(id);
              }}
            />
          ) : null}

          {templateError ? (
            <Banner kind="inline" status="error">
              {templateError}
            </Banner>
          ) : null}
        </Stack>
      </Block>

      {selectedOption ? (
        <Flex
          align="center"
          justify="end"
          className="shrink-0 gap-4 border-t border-base bg-surface-base px-10 py-4"
        >
          {!canContinue && !isSettingUp && BLOCKED_HINT[selectedOption.id] ? (
            <Text kind="body/regular/sm" className="text-secondary">
              {BLOCKED_HINT[selectedOption.id]}
            </Text>
          ) : null}
          <LoadingButton
            color="brand"
            kind="primary"
            loading={isSettingUp}
            onClick={() => void handleContinue()}
            disabled={!canContinue}
          >
            {isSettingUp ? (
              statusLabel
            ) : (
              <>
                Continue
                <ArrowRight size={16} aria-hidden />
              </>
            )}
          </LoadingButton>
        </Flex>
      ) : null}
    </Stack>
  );
};
