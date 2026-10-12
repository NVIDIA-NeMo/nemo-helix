// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Banner, Flex, Spinner, Stack } from '@nvidia/foundations-react-core';
import {
  START_OPTIONS,
  TEMPLATE_GROUP_TITLE,
} from '@studio/components/CreateCustomizationStart/constants';
import { DeleteSavedTemplate } from '@studio/components/CreateCustomizationStart/DeleteSavedTemplate';
import { TemplateConflictBanner } from '@studio/components/CreateCustomizationStart/TemplateConflictBanner';
import type {
  CreateCustomizationStartProps,
  StartOptionId,
} from '@studio/components/CreateCustomizationStart/types';
import { useSavedTemplates } from '@studio/components/CreateCustomizationStart/useSavedTemplates';
import {
  useTemplateSetup,
  type ConflictResolution,
} from '@studio/components/CreateCustomizationStart/useTemplateSetup';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';
import {
  CUSTOMIZATION_TEMPLATES,
  type CustomizationTemplate,
} from '@studio/constants/customizationTemplates';
import { getFilesetRoute } from '@studio/routes/utils';
import { toCustomizationBackend } from '@studio/util/customizationBackend';
import {
  templateToFormFields,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import { Box, Bookmark } from 'lucide-react';
import { lazy, Suspense, useEffect, useMemo, useRef, useState, type FC } from 'react';
import { Link } from 'react-router';

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

/** Why Continue is unavailable, shown next to the disabled button. */
const BLOCKED_HINT: Partial<Record<StartOptionId, string>> = {
  template: 'Pick a recipe to continue.',
  ai: 'Draft settings that pass the checks to continue.',
};

/** Templates are the middle rung, and the likeliest way in, so the page opens on them. */
const DEFAULT_OPTION: StartOptionId = 'template';

export const CreateCustomizationStart: FC<CreateCustomizationStartProps> = ({
  workspace,
  onContinue,
}) => {
  const [selectedId, setSelectedId] = useState<StartOptionId>(DEFAULT_OPTION);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  // Set only once a generated draft validates, so Continue can never load a broken config.
  const [draftValues, setDraftValues] = useState<CustomizationFormFields | null>(null);

  const toast = useToast();
  const {
    run: runTemplateSetup,
    statusLabel,
    error: templateError,
    conflict,
    clearConflict,
  } = useTemplateSetup(workspace);
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
    if (selectedId === 'ai') {
      if (draftValues) onContinue({ optionId: 'ai', initialValues: draftValues });
      return;
    }
    // Names a model and dataset the workspace already has, so nothing to provision.
    if (selectedSaved) {
      const initialValues = templateToFormFields(selectedSaved);
      if (initialValues) onContinue({ optionId: 'template', initialValues });
      return;
    }
    if (!selectedTemplate) return;
    await provision(selectedTemplate);
  };

  /**
   * Registering the model and loading the dataset has to finish before the form can
   * reference them, so it happens here rather than on the next screen.
   *
   * `resolution` is set when the user is answering a fileset conflict rather than starting
   * the recipe fresh.
   */
  const provision = async (template: CustomizationTemplate, resolution?: ConflictResolution) => {
    const result = await runTemplateSetup(template, resolution);
    // Provisioning spans a render, and the page is locked throughout, but only hand over
    // values that still match what is selected, and only if there is still a picker to
    // hand them over from.
    if (!result || !mounted.current || selectedTemplateId !== template.id) return;

    // Which fileset the job trains on is the whole point of the recipe, so a reuse is
    // stated rather than left silent. It goes to a toast because handing the values over
    // navigates away from this page immediately.
    if (result.reusedFilesetRef) {
      toast.info(
        <span>
          Reused the existing dataset{' '}
          <Link
            to={getFilesetRoute(workspace, result.reusedFilesetRef)}
            className="text-primary underline"
          >
            {result.reusedFilesetRef}
          </Link>{' '}
          instead of downloading it again.
        </span>,
        { durationMs: 10_000 }
      );
    }

    onContinue({ optionId: 'template', initialValues: result.values });
  };

  return (
    <StartPage
      heading="Fine-tune a Model"
      headingDescription="Train a model on your own data. Describe what you need and let AI draft the settings, pick a ready-made recipe, or set everything up yourself."
      options={START_OPTIONS}
      value={selectedId}
      onChange={(id) => {
        setSelectedId(id as StartOptionId);
        setSelectedTemplateId(null);
        clearConflict();
        setDraftValues(null);
      }}
      // Provisioning registers models and uploads a dataset, which takes long enough that
      // the cards would stay clickable behind the disabled Continue. Moving the selection
      // then would leave a finished setup pointing at something else.
      disabled={isSettingUp}
      canContinue={
        !isSettingUp &&
        (selectedId === 'scratch' ||
          (selectedId === 'ai' ? draftValues !== null : selectedTemplateId !== null))
      }
      continueLabel={isSettingUp ? statusLabel : 'Continue'}
      continueLoading={isSettingUp}
      onContinue={() => void handleContinue()}
      blockedHint={BLOCKED_HINT[selectedId]}
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
        selectedId === 'ai' ? (
          <Suspense fallback={panelFallback}>
            <DescribeWithAiPanel workspace={workspace} onDraft={setDraftValues} />
          </Suspense>
        ) : selectedId === 'template' ? (
          <Stack gap="density-2xl" className="w-full">
            <TemplateGroups
              groups={templateGroups}
              value={selectedTemplateId}
              // A conflict names one recipe's fileset; moving off that recipe retires it.
              onChange={(id) => {
                setSelectedTemplateId(id);
                clearConflict();
              }}
              disabled={isSettingUp}
            />
          </Stack>
        ) : null
      }
      slotBanner={
        conflict ? (
          <TemplateConflictBanner
            conflict={conflict}
            busy={isSettingUp}
            onRename={() =>
              selectedTemplate && void provision(selectedTemplate, { action: 'rename', conflict })
            }
            onReplace={() =>
              selectedTemplate && void provision(selectedTemplate, { action: 'replace', conflict })
            }
          />
        ) : templateError ? (
          <Banner kind="inline" status="error">
            {templateError}
          </Banner>
        ) : null
      }
    />
  );
};
