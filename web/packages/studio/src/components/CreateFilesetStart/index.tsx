// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { START_OPTIONS } from '@studio/components/CreateFilesetStart/constants';
import { StartOptionDetail } from '@studio/components/CreateFilesetStart/StartOptionDetail';
import type {
  CreateFilesetStartProps,
  StartOptionId,
} from '@studio/components/CreateFilesetStart/types';
import { StartPage } from '@studio/components/StartOptions/StartPage';
import { useCallback, useState, type FC } from 'react';

/** Why Continue is unavailable, shown next to the disabled button. */
const BLOCKED_HINT: Partial<Record<StartOptionId, string>> = {
  template: 'Pick a recipe to continue.',
  ai: 'Generate a valid config to continue.',
};

/** Templates are the middle rung, and the likeliest way in, so the page opens on them. */
const DEFAULT_OPTION: StartOptionId = 'template';

export const CreateFilesetStart: FC<CreateFilesetStartProps> = ({ workspace, onContinue }) => {
  const [selectedId, setSelectedId] = useState<StartOptionId>(DEFAULT_OPTION);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  // Set only once a generated draft validates, so Continue can never load a broken config.
  const [generatedJobRequest, setGeneratedJobRequest] = useState<DataDesignerJobRequest | null>(
    null
  );
  const selectedOption = START_OPTIONS.find((option) => option.id === selectedId) ?? null;

  const selectOption = (optionId: string) => {
    setSelectedId(optionId as StartOptionId);
    setSelectedTemplateId(null);
    setGeneratedJobRequest(null);
  };

  // Identity-stable so it can be a dependency of the AI panel's generate callback.
  const handleValidConfig = useCallback(
    (jobRequest: DataDesignerJobRequest | null) => setGeneratedJobRequest(jobRequest),
    []
  );

  // A chosen tile, plus that option's own payload: a recipe for "template", a validated
  // config for "ai". "scratch" needs nothing else.
  const canContinue =
    selectedOption !== null &&
    (selectedOption.id !== 'template' || selectedTemplateId !== null) &&
    (selectedOption.id !== 'ai' || generatedJobRequest !== null);

  const handleContinue = () => {
    if (!selectedOption) return;
    if (selectedOption.id === 'template' && selectedTemplateId) {
      onContinue({ optionId: 'template', templateId: selectedTemplateId });
    } else if (selectedOption.id === 'ai' && generatedJobRequest) {
      onContinue({ optionId: 'ai', jobRequest: generatedJobRequest });
    } else if (selectedOption.id === 'scratch') {
      onContinue({ optionId: 'scratch' });
    }
  };

  return (
    <StartPage
      heading="Create a fileset"
      headingDescription="Generate synthetic data visually — no JSON to write. Start from a template, or describe what you need and let AI lay out the columns."
      options={START_OPTIONS}
      value={selectedId}
      onChange={selectOption}
      canContinue={canContinue}
      onContinue={handleContinue}
      blockedHint={BLOCKED_HINT[selectedId]}
      slotDetail={
        selectedOption ? (
          <StartOptionDetail
            option={selectedOption}
            selectedTemplateId={selectedTemplateId}
            onSelectTemplate={setSelectedTemplateId}
            workspace={workspace}
            onValidConfig={handleValidConfig}
          />
        ) : null
      }
    />
  );
};
