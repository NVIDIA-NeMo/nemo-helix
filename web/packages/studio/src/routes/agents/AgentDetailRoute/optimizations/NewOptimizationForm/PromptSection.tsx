// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2';
import { useAgentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import {
  Banner,
  Button,
  Flex,
  FormField,
  SegmentedControl,
  Skeleton,
  Stack,
  Switch,
  Text,
  TextArea,
} from '@nvidia/foundations-react-core';
import { Add, Close } from '@nvidia/react-gui-icons/icons';
import {
  type OptimizationFormValues,
  sweepableVersions,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import { agentSystemPrompt } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizeConfig';
import {
  budgetById,
  PROMPT_MODES,
  promptModeById,
  type PromptModeId,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';
import { type FC, useEffect, useRef } from 'react';
import { useFormContext } from 'react-hook-form';

/** The fewest versions worth sweeping; one is just a fixed override. */
const MIN_VERSIONS = 2;

export interface PromptSectionProps {
  workspace: string;
  agentName?: string;
}

/**
 * Whether to tune the agent's system prompt, and how: let a model rewrite it after the sweep, or
 * sweep versions the user writes. Off by default, since the rewrite multiplies the run's cost;
 * its fields stay hidden until then, so the sweep alone still reads as three questions.
 */
export const PromptSection: FC<PromptSectionProps> = ({ workspace, agentName }) => {
  const {
    watch,
    setValue,
    formState: { isSubmitting, errors, touchedFields },
  } = useFormContext<OptimizationFormValues>();
  const prompt = watch('prompt');
  const trials = budgetById(watch('budget')).trials;

  const { data: agent, isLoading: isAgentLoading } = useAgentsGetAgent(workspace, agentName ?? '', {
    query: { enabled: !!agentName && prompt.enabled },
  });
  const currentPrompt = agentSystemPrompt(agent?.config);

  // Start the user from the prompt the agent runs today, once, and only into an untouched list.
  const seeded = useRef(false);
  useEffect(() => {
    if (seeded.current || currentPrompt === undefined) return;
    seeded.current = true;
    if (prompt.versions.every((version) => !version.trim())) {
      setValue('prompt.versions', [currentPrompt, ...prompt.versions.slice(1)]);
    }
  }, [currentPrompt, prompt.versions, setValue]);

  const purposeError = touchedFields.prompt?.purpose ? errors.prompt?.purpose?.message : undefined;
  const modelError = touchedFields.prompt?.model ? errors.prompt?.model?.message : undefined;

  const setVersions = (versions: string[]) =>
    setValue('prompt.versions', versions, { shouldValidate: true });
  const sweepable = sweepableVersions(prompt.versions).length;

  return (
    <Stack gap="density-xl" className="w-full">
      <Switch
        checked={prompt.enabled}
        disabled={isSubmitting}
        onCheckedChange={(checked) => setValue('prompt.enabled', checked, { shouldValidate: true })}
        slotLabel="Optimize the system prompt"
      />

      {prompt.enabled && (
        <>
          <Stack gap="density-sm">
            <SegmentedControl
              aria-label="How to optimize the prompt"
              className="w-full"
              value={prompt.mode}
              // SegmentedControl has no disabled state, so a submitting form ignores the change.
              onValueChange={(value) => {
                if (isSubmitting) return;
                setValue('prompt.mode', value as PromptModeId, { shouldValidate: true });
              }}
              items={PROMPT_MODES.map((mode) => ({ value: mode.id, children: mode.title }))}
            />
            <Text kind="body/regular/sm" color="secondary">
              {promptModeById(prompt.mode).description}
            </Text>
          </Stack>

          {prompt.mode === 'rewrite' && (
            <>
              <FormField
                slotLabel="What should the prompt achieve?"
                slotError={purposeError}
                status={purposeError ? 'error' : undefined}
              >
                <TextArea
                  value={prompt.purpose}
                  rows={3}
                  disabled={isSubmitting}
                  status={purposeError ? 'error' : undefined}
                  placeholder="Answer support questions accurately and cite the policy that applies."
                  onValueChange={(value) =>
                    setValue('prompt.purpose', value, { shouldValidate: true, shouldTouch: true })
                  }
                />
              </FormField>

              <FormField
                slotLabel="Rewrite with"
                slotError={modelError}
                status={modelError ? 'error' : undefined}
              >
                <WorkspaceModelSelect
                  workspace={workspace}
                  value={prompt.model ? { model: prompt.model } : null}
                  onValueChange={({ model }) =>
                    setValue('prompt.model', model, { shouldValidate: true, shouldTouch: true })
                  }
                  placeholder="Select a model"
                  disabled={isSubmitting}
                  hideAdapters
                  fullWidth
                  aria-label="Prompt rewriter model"
                />
              </FormField>

              {agent && currentPrompt === undefined && (
                <Banner kind="inline" status="warning">
                  {agentName} has no system prompt to rewrite. Add one to the agent, or try your own
                  versions instead.
                </Banner>
              )}
            </>
          )}

          {prompt.mode === 'versions' && isAgentLoading && (
            <Stack gap="density-lg" aria-busy aria-label="Loading the current prompt">
              {Array.from({ length: MIN_VERSIONS }, (_, index) => (
                <Skeleton key={index} className="h-24 w-full" />
              ))}
            </Stack>
          )}

          {prompt.mode === 'versions' && !isAgentLoading && (
            <Stack gap="density-lg">
              {prompt.versions.map((version, index) => (
                <FormField
                  // Versions have no identity beyond their position, and are only ever removed
                  // by it, so the index is a stable key here.
                  key={index}
                  slotLabel={
                    <Flex align="center" justify="between" className="w-full">
                      <span>
                        Version {index + 1}
                        {currentPrompt !== undefined && version === currentPrompt
                          ? ' · current prompt'
                          : ''}
                      </span>
                      {prompt.versions.length > MIN_VERSIONS && (
                        <Button
                          type="button"
                          kind="tertiary"
                          size="small"
                          disabled={isSubmitting}
                          aria-label={`Remove version ${index + 1}`}
                          onClick={() =>
                            setVersions(prompt.versions.filter((_, other) => other !== index))
                          }
                        >
                          <Close size="16" />
                        </Button>
                      )}
                    </Flex>
                  }
                >
                  <TextArea
                    value={version}
                    rows={4}
                    disabled={isSubmitting}
                    placeholder="You are a support agent…"
                    attributes={{ TextAreaElement: { 'aria-label': `Version ${index + 1}` } }}
                    onValueChange={(value) =>
                      setVersions(
                        prompt.versions.map((other, position) =>
                          position === index ? value : other
                        )
                      )
                    }
                  />
                </FormField>
              ))}

              <Button
                type="button"
                kind="secondary"
                className="w-fit"
                disabled={isSubmitting}
                onClick={() => setVersions([...prompt.versions, ''])}
              >
                <Add size="16" aria-hidden />
                Add a version
              </Button>

              {sweepable > trials && (
                <Banner kind="inline" status="warning">
                  {sweepable} versions but only {trials} trials, so some versions may never run.
                  Pick a bigger budget to give each one a chance.
                </Banner>
              )}
            </Stack>
          )}
        </>
      )}
    </Stack>
  );
};
