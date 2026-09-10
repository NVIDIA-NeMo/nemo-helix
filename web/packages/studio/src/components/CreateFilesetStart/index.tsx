// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import {
  Block,
  Button,
  Flex,
  Grid,
  GridItem,
  PageHeader,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { START_OPTIONS } from '@studio/components/CreateFilesetStart/constants';
import { StartOptionDetail } from '@studio/components/CreateFilesetStart/StartOptionDetail';
import type {
  CreateFilesetStartProps,
  StartOptionId,
} from '@studio/components/CreateFilesetStart/types';
import { StartOptionCard } from '@studio/components/StartOptions/StartOptionCard';
import { ArrowRight } from 'lucide-react';
import { useCallback, useState, type FC } from 'react';

/** Why Continue is unavailable, shown next to the disabled button. */
const BLOCKED_HINT: Partial<Record<StartOptionId, string>> = {
  template: 'Pick a recipe to continue.',
  ai: 'Generate a valid config to continue.',
};

export const CreateFilesetStart: FC<CreateFilesetStartProps> = ({ workspace, onContinue }) => {
  const [selectedId, setSelectedId] = useState<StartOptionId | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  // Set only once a generated draft passes validation, so Continue can never load a broken config.
  const [generatedJobRequest, setGeneratedJobRequest] = useState<DataDesignerJobRequest | null>(
    null
  );
  const selectedOption = START_OPTIONS.find((option) => option.id === selectedId) ?? null;

  const selectOption = (optionId: StartOptionId) => {
    setSelectedId(optionId);
    setSelectedTemplateId(null);
    setGeneratedJobRequest(null);
  };

  // Identity-stable so it can be a dependency of the AI panel's generate callback.
  const handleValidConfig = useCallback(
    (jobRequest: DataDesignerJobRequest | null) => setGeneratedJobRequest(jobRequest),
    []
  );

  // Ready to continue once a tile is chosen — plus that option's own payload: a template card
  // for "template", a validated config for "ai".
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
    <Stack className="h-full">
      <Block className="flex-1 overflow-auto">
        <Stack gap="density-2xl" padding="density-2xl">
          <PageHeader
            slotHeading="Create a fileset"
            slotDescription="Generate synthetic data visually — no JSON to write. Start from a template, clone a fileset you already built, or describe what you need and let AI lay out the columns."
          />

          <Stack gap="density-md">
            <Text kind="label/bold/sm" className="text-secondary">
              How do you want to start?
            </Text>
            <Grid colMinWidth="200px" gap="density-md">
              {START_OPTIONS.map((option) => (
                <GridItem key={option.id}>
                  <StartOptionCard
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
              selectedTemplateId={selectedTemplateId}
              onSelectTemplate={setSelectedTemplateId}
              workspace={workspace}
              onValidConfig={handleValidConfig}
            />
          ) : null}
        </Stack>
      </Block>

      {selectedOption ? (
        <Flex
          align="center"
          justify="end"
          className="shrink-0 gap-4 border-t border-base bg-surface-base px-10 py-4"
        >
          {!canContinue && BLOCKED_HINT[selectedOption.id] ? (
            <Text kind="body/regular/sm" className="text-secondary">
              {BLOCKED_HINT[selectedOption.id]}
            </Text>
          ) : null}
          <Button color="brand" kind="primary" onClick={handleContinue} disabled={!canContinue}>
            Continue
            <ArrowRight size={16} aria-hidden />
          </Button>
        </Flex>
      ) : null}
    </Stack>
  );
};
