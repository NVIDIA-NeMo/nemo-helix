// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { START_OPTIONS } from '@studio/components/CreateFilesetStart/constants';
import { DescribeWithAiPanel } from '@studio/components/CreateFilesetStart/DescribeWithAiPanel';
import { buildTemplateGroups } from '@studio/components/CreateFilesetStart/templateGroups';
import type { CreateFilesetStartProps } from '@studio/components/CreateFilesetStart/types';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { StartSubPage } from '@studio/components/StartOptions/StartSubPage';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import { useCallback, useMemo, useState, type FC } from 'react';

const AI_OPTION = START_OPTIONS.find((option) => option.id === 'ai');

export const CreateFilesetStart: FC<CreateFilesetStartProps> = ({ workspace, onContinue }) => {
  const [isDescribing, setIsDescribing] = useState(false);
  // Set only once a generated draft validates, so Continue can never load a broken config.
  const [generatedJobRequest, setGeneratedJobRequest] = useState<DataDesignerJobRequest | null>(
    null
  );
  const templateGroups = useMemo(buildTemplateGroups, []);

  // Identity-stable so it can be a dependency of the AI panel's generate callback.
  const handleValidConfig = useCallback(
    (jobRequest: DataDesignerJobRequest | null) => setGeneratedJobRequest(jobRequest),
    []
  );

  if (isDescribing && AI_OPTION) {
    return (
      <StartSubPage
        heading={AI_OPTION.title}
        headingDescription={AI_OPTION.description}
        onBack={() => {
          setIsDescribing(false);
          setGeneratedJobRequest(null);
        }}
        canContinue={generatedJobRequest !== null}
        onContinue={() => {
          if (generatedJobRequest) onContinue({ optionId: 'ai', jobRequest: generatedJobRequest });
        }}
        blockedHint="Generate a valid config to continue."
      >
        <DescribeWithAiPanel workspace={workspace} onValidConfig={handleValidConfig} />
      </StartSubPage>
    );
  }

  return (
    <StartPage
      heading="Create a fileset"
      headingDescription="Generate synthetic data visually — no JSON to write. Start from a template, or describe what you need and let AI lay out the columns."
      options={START_OPTIONS}
      // Templates are the likeliest way in, so their cards are always the panel here.
      value="template"
      onChange={(id) => {
        if (id === 'ai') setIsDescribing(true);
        if (id === 'scratch') onContinue({ optionId: 'scratch' });
      }}
      slotDetail={
        <TemplateGroups
          groups={templateGroups}
          onSelect={(templateId) => onContinue({ optionId: 'template', templateId })}
        />
      }
    />
  );
};
