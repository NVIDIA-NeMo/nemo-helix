// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Stack } from '@nvidia/foundations-react-core';
import {
  START_OPTIONS,
  TEMPLATE_GROUP_TITLE,
} from '@studio/components/CreateCustomizationStart/constants';
import { DeleteSavedTemplate } from '@studio/components/CreateCustomizationStart/DeleteSavedTemplate';
import type {
  CreateCustomizationStartProps,
  StartOptionId,
} from '@studio/components/CreateCustomizationStart/types';
import { useSavedTemplates } from '@studio/components/CreateCustomizationStart/useSavedTemplates';
import { useTemplateSetup } from '@studio/components/CreateCustomizationStart/useTemplateSetup';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';
import { CUSTOMIZATION_TEMPLATES } from '@studio/constants/customizationTemplates';
import { toCustomizationBackend } from '@studio/util/customizationBackend';
import { templateToFormFields } from '@studio/util/forms/customization';
import { Box, Bookmark } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type FC } from 'react';

/** Namespaces saved-template ids so they cannot collide with a curated recipe's id. */
const SAVED_PREFIX = 'saved:';

const savedTemplateKey = (template: { name?: string; id: string }) =>
  `${SAVED_PREFIX}${template.name ?? template.id}`;

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

  const {
    data: saved,
    isLoading: savedLoading,
    refetch: refetchSaved,
  } = useSavedTemplates(workspace);

  // Setup runs long enough that leaving the page part-way through is a realistic move, and
  // nothing here blocks it. The promise resolves regardless of whether this is still on
  // screen, and `onContinue` navigates — so without this, finishing setup would yank the
  // user to the form from wherever they had gone.
  // Set on the way in as well as cleared on the way out: StrictMode runs an effect, its
  // cleanup, then the effect again, so a cleanup-only version latches to false on mount in
  // development and never hands anything over.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // A backend the form has no arm for cannot seed it, so it is not offered.
  const savedTemplates = useMemo(
    () => (saved ?? []).filter((template) => toCustomizationBackend(template.backend)),
    [saved]
  );

  const templateGroups = useMemo<StartTemplateGroup[]>(
    () => [
      {
        id: 'saved-templates',
        title: 'Saved templates',
        loading: savedLoading,
        accent: 'var(--text-color-accent-teal)',
        templates: savedTemplates.map((template) => ({
          id: savedTemplateKey(template),
          name: template.name ?? template.id,
          description: template.description || 'Saved from an earlier job.',
          icon: Bookmark,
        })),
      },
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
    [savedTemplates, savedLoading]
  );

  const selectedTemplate =
    CUSTOMIZATION_TEMPLATES.find((template) => template.id === selectedTemplateId) ?? null;

  const selectedSaved =
    savedTemplates.find((template) => savedTemplateKey(template) === selectedTemplateId) ?? null;

  const handleContinue = async () => {
    if (selectedId === 'scratch') {
      onContinue({ optionId: 'scratch' });
      return;
    }
    // Names a model and dataset the workspace already has, so nothing to provision.
    if (selectedSaved) {
      const initialValues = templateToFormFields(selectedSaved);
      if (initialValues) onContinue({ optionId: 'template', initialValues });
      return;
    }
    if (!selectedTemplate) return;

    // Registering the model and loading the dataset has to finish before the form can
    // reference them, so it happens here rather than on the next screen.
    const initialValues = await runTemplateSetup(selectedTemplate);
    // Provisioning spans a render, and the page is locked throughout, but only hand over
    // values that still match what is selected, and only if there is still a picker to
    // hand them over from.
    if (initialValues && mounted.current && selectedTemplateId === selectedTemplate.id) {
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
      slotFooterStart={
        selectedSaved ? (
          <DeleteSavedTemplate
            workspace={workspace}
            template={selectedSaved}
            onDeleted={() => {
              setSelectedTemplateId(null);
              void refetchSaved();
            }}
          />
        ) : null
      }
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
