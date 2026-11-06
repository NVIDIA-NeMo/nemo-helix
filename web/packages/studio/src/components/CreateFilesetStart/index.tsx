// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { START_OPTIONS } from '@studio/components/CreateFilesetStart/constants';
import { DescribeWithAiPanel } from '@studio/components/CreateFilesetStart/DescribeWithAiPanel';
import { PersonaTestSetPanel } from '@studio/components/CreateFilesetStart/PersonaTestSetPanel';
import { buildTemplateGroups } from '@studio/components/CreateFilesetStart/templateGroups';
import type { CreateFilesetStartProps } from '@studio/components/CreateFilesetStart/types';
import { useHasPersonaAgents } from '@studio/components/CreateFilesetStart/useHasPersonaAgents';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { StartSubPage } from '@studio/components/StartOptions/StartSubPage';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import { useCallback, useMemo, useState, type FC } from 'react';

const AI_OPTION = START_OPTIONS.find((option) => option.id === 'ai');
const PERSONA_OPTION = START_OPTIONS.find((option) => option.id === 'persona');

export const CreateFilesetStart: FC<CreateFilesetStartProps> = ({ workspace, onContinue }) => {
  const [step, setStep] = useState<'ai' | 'persona' | null>(null);
  // Set only once a generated draft validates, so Continue can never load a broken config.
  const [generatedJobRequest, setGeneratedJobRequest] = useState<DataDesignerJobRequest | null>(
    null
  );
  const templateGroups = useMemo(buildTemplateGroups, []);
  // Personas write messages for a chosen agent, so the tile is a dead end without one.
  // Only a confirmed-empty list disables it; a failed load stays reachable.
  const noAgents = useHasPersonaAgents(workspace) === false;
  const options = useMemo(
    () =>
      START_OPTIONS.map((option) =>
        option.id === 'persona' && noAgents
          ? {
              ...option,
              enabled: false,
              disabledReason: 'Personas write test messages for an agent. Create an agent first.',
            }
          : option
      ),
    [noAgents]
  );

  // Identity-stable so it can be a dependency of the AI panel's generate callback.
  const handleValidConfig = useCallback(
    (jobRequest: DataDesignerJobRequest | null) => setGeneratedJobRequest(jobRequest),
    []
  );

  const stepOption = step === 'persona' ? PERSONA_OPTION : step === 'ai' ? AI_OPTION : undefined;
  if (step && stepOption) {
    return (
      <StartSubPage
        heading={stepOption.title}
        headingDescription={stepOption.description}
        onBack={() => {
          setStep(null);
          setGeneratedJobRequest(null);
        }}
        canContinue={generatedJobRequest !== null}
        onContinue={() => {
          if (generatedJobRequest) onContinue({ optionId: step, jobRequest: generatedJobRequest });
        }}
        blockedHint={
          step === 'ai'
            ? 'Generate a valid config to continue.'
            : 'Select an agent and set up the persona dataset to continue.'
        }
      >
        {step === 'ai' ? (
          <DescribeWithAiPanel workspace={workspace} onValidConfig={handleValidConfig} />
        ) : (
          <PersonaTestSetPanel workspace={workspace} onValidConfig={handleValidConfig} />
        )}
      </StartSubPage>
    );
  }

  return (
    <StartPage
      heading="Create a fileset"
      headingDescription="Generate synthetic data visually — no JSON to write. Start from a template, or describe what you need and let AI lay out the columns."
      options={options}
      // Templates are the likeliest way in, so their cards are always the panel here.
      value="template"
      onChange={(id) => {
        if (id === 'ai' || id === 'persona') setStep(id);
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
