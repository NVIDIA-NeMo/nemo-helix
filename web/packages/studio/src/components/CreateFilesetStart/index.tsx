// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AiDraftStep } from '@studio/components/CreateFilesetStart/AiDraftStep';
import {
  OTHER_SECTION,
  START_OPTIONS,
  TEMPLATES_TAG,
  TEMPLATE_SECTIONS,
} from '@studio/components/CreateFilesetStart/constants';
import { FILESET_TEMPLATES } from '@studio/components/CreateFilesetStart/templates';
import type { CreateFilesetStartProps } from '@studio/components/CreateFilesetStart/types';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';
import { useMemo, useState, type FC } from 'react';

/**
 * Groups the templates into the sections the page renders, in the order
 * {@link TEMPLATE_SECTIONS} names them. A template's own tag decides its section, and the
 * ones whose tag names no section collect under "Other" rather than each earning a
 * heading of one.
 */
const buildTemplateGroups = (): StartTemplateGroup[] => {
  const sections = new Map<string, StartTemplateGroup>(
    [...TEMPLATE_SECTIONS, OTHER_SECTION].map((title) => [
      title,
      { id: title.toLowerCase(), title, templates: [] },
    ])
  );

  for (const template of FILESET_TEMPLATES) {
    const section = sections.get(template.tag.label) ?? sections.get(OTHER_SECTION);
    section?.templates.push({
      id: template.id,
      name: template.title,
      description: template.description,
      icon: template.icon,
    });
  }

  return [...sections.values()].filter((section) => section.templates.length > 0);
};

export const CreateFilesetStart: FC<CreateFilesetStartProps> = ({ workspace, onContinue }) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // "Describe with AI" is the only option that asks for anything else before the canvas.
  const [draftingWithAi, setDraftingWithAi] = useState(false);

  const templateGroups = useMemo(buildTemplateGroups, []);

  const isTemplate = FILESET_TEMPLATES.some((template) => template.id === selectedId);

  const handleContinue = () => {
    if (selectedId === 'scratch') {
      onContinue({ optionId: 'scratch' });
      return;
    }
    if (selectedId === 'ai') {
      setDraftingWithAi(true);
      return;
    }
    if (isTemplate && selectedId) {
      onContinue({ optionId: 'template', templateId: selectedId });
    }
  };

  if (draftingWithAi) {
    return (
      <AiDraftStep
        workspace={workspace}
        onBack={() => setDraftingWithAi(false)}
        onContinue={(jobRequest) => onContinue({ optionId: 'ai', jobRequest })}
      />
    );
  }

  return (
    <StartPage
      heading="Create a fileset"
      headingDescription="Generate synthetic data visually — no JSON to write. Start from a template, clone a fileset you already built, or describe what you need and let AI lay out the columns."
      options={START_OPTIONS}
      templateGroups={templateGroups}
      templatesTag={TEMPLATES_TAG}
      value={selectedId}
      onChange={setSelectedId}
      canContinue={selectedId !== null}
      onContinue={handleContinue}
    />
  );
};
