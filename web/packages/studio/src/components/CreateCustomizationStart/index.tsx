// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Flex, Spinner } from '@nvidia/foundations-react-core';
import {
  START_OPTIONS,
  TEMPLATE_GROUP_TITLE,
} from '@studio/components/CreateCustomizationStart/constants';
import { DeleteSavedTemplate } from '@studio/components/CreateCustomizationStart/DeleteSavedTemplate';
import type { CreateCustomizationStartProps } from '@studio/components/CreateCustomizationStart/types';
import { useSavedTemplates } from '@studio/components/CreateCustomizationStart/useSavedTemplates';
import { useTemplateSetup } from '@studio/components/CreateCustomizationStart/useTemplateSetup';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { StartSubPage } from '@studio/components/StartOptions/StartSubPage';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';
import { CUSTOMIZATION_TEMPLATES } from '@studio/constants/customizationTemplates';
import { toCustomizationBackend } from '@studio/util/customizationBackend';
import {
  templateToFormFields,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import { Box, Bookmark } from 'lucide-react';
import { lazy, Suspense, useEffect, useMemo, useRef, useState, type FC } from 'react';

// Inlines the customizer skill's references (~75 KB), so it loads only once AI is picked.
const DescribeWithAiPanel = lazy(() =>
  import('@studio/components/CreateCustomizationStart/DescribeWithAiPanel').then((module) => ({
    default: module.DescribeWithAiPanel,
  }))
);

const panelFallback = (
  <Flex align="center" justify="center" className="h-64">
    <Spinner size="medium" aria-label="Loading..." />
  </Flex>
);

/** Namespaces saved-template ids so they cannot collide with a curated recipe's id. */
const SAVED_PREFIX = 'saved:';

const savedTemplateKey = (template: { name?: string; id: string }) =>
  `${SAVED_PREFIX}${template.name ?? template.id}`;

const AI_OPTION = START_OPTIONS.find((option) => option.id === 'ai');

export const CreateCustomizationStart: FC<CreateCustomizationStartProps> = ({
  workspace,
  onContinue,
}) => {
  const [isDescribing, setIsDescribing] = useState(false);
  const [pendingTemplateId, setPendingTemplateId] = useState<string | null>(null);
  // Set only once a generated draft validates, so Continue can never load a broken config.
  const [draftValues, setDraftValues] = useState<CustomizationFormFields | null>(null);

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
          action: (
            <DeleteSavedTemplate
              workspace={workspace}
              template={template}
              onDeleted={() => void refetchSaved()}
              disabled={isSettingUp}
            />
          ),
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
    [savedTemplates, savedLoading, workspace, refetchSaved, isSettingUp]
  );

  const startFromTemplate = async (templateId: string) => {
    // Names a model and dataset the workspace already has, so nothing to provision.
    const savedTemplate = savedTemplates.find(
      (template) => savedTemplateKey(template) === templateId
    );
    if (savedTemplate) {
      const initialValues = templateToFormFields(savedTemplate);
      if (initialValues) onContinue({ optionId: 'template', initialValues });
      return;
    }
    const recipe = CUSTOMIZATION_TEMPLATES.find((template) => template.id === templateId);
    if (!recipe) return;

    // Registering the model and loading the dataset has to finish before the form can
    // reference them, so it happens here, with every card locked until it does.
    setPendingTemplateId(templateId);
    const initialValues = await runTemplateSetup(recipe);
    // Only hand over values if there is still a picker to hand them over from.
    if (!mounted.current) return;
    if (initialValues) {
      onContinue({ optionId: 'template', initialValues });
    } else {
      setPendingTemplateId(null);
    }
  };

  if (isDescribing && AI_OPTION) {
    return (
      <StartSubPage
        heading={AI_OPTION.title}
        headingDescription={AI_OPTION.description}
        onBack={() => {
          setIsDescribing(false);
          setDraftValues(null);
        }}
        canContinue={draftValues !== null}
        onContinue={() => {
          if (draftValues) onContinue({ optionId: 'ai', initialValues: draftValues });
        }}
        blockedHint="Draft settings that pass the checks to continue."
      >
        <Suspense fallback={panelFallback}>
          <DescribeWithAiPanel workspace={workspace} onDraft={setDraftValues} />
        </Suspense>
      </StartSubPage>
    );
  }

  return (
    <StartPage
      heading="Fine-tune a Model"
      headingDescription="Train a model on your own data. Describe what you need and let AI draft the settings, pick a ready-made recipe, or set everything up yourself."
      options={START_OPTIONS}
      // Templates are the likeliest way in, so their cards are always the panel here.
      value="template"
      onChange={(id) => {
        if (id === 'ai') setIsDescribing(true);
        if (id === 'scratch') onContinue({ optionId: 'scratch' });
      }}
      // Provisioning registers models and uploads a dataset. Every card stays locked until
      // it settles, so a finished setup can never hand over something the user moved off.
      disabled={isSettingUp}
      slotDetail={
        <TemplateGroups
          groups={templateGroups}
          onSelect={(id) => void startFromTemplate(id)}
          pendingId={pendingTemplateId}
          pendingLabel={statusLabel}
          disabled={isSettingUp}
        />
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
