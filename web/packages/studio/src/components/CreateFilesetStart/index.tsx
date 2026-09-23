// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AiDraftStep } from '@studio/components/CreateFilesetStart/AiDraftStep';
import { START_OPTIONS, TEMPLATES_TAG } from '@studio/components/CreateFilesetStart/constants';
import { buildTemplateGroups } from '@studio/components/CreateFilesetStart/templateGroups';
import { FILESET_TEMPLATES } from '@studio/components/CreateFilesetStart/templates';
import type { CreateFilesetStartProps } from '@studio/components/CreateFilesetStart/types';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { useMemo, useState, type FC } from 'react';

export const CreateFilesetStart: FC<CreateFilesetStartProps> = ({ workspace, onContinue }) => {
  // "Describe with AI" is the only option that asks for anything else before the canvas.
  const [draftingWithAi, setDraftingWithAi] = useState(false);

  const templateGroups = useMemo(buildTemplateGroups, []);

  const handleSelect = (id: string) => {
    if (id === 'scratch') {
      onContinue({ optionId: 'scratch' });
      return;
    }
    if (id === 'ai') {
      setDraftingWithAi(true);
      return;
    }
    if (FILESET_TEMPLATES.some((template) => template.id === id)) {
      onContinue({ optionId: 'template', templateId: id });
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
      headingDescription="Generate synthetic data visually — no JSON to write. Start from a template, or describe what you need and let AI lay out the columns."
      options={START_OPTIONS}
      templateGroups={templateGroups}
      templatesTag={TEMPLATES_TAG}
      onSelect={handleSelect}
    />
  );
};
