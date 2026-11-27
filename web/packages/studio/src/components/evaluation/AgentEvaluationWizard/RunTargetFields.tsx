// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { useListExperiments } from '@nemo/sdk/generated/platform/experiments';
import {
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
} from '@nvidia/foundations-react-core';
import { useRunNameCheck } from '@studio/components/evaluation/AgentEvaluationWizard/runTarget';
import type { RunTargetFormFields } from '@studio/components/evaluation/AgentEvaluationWizard/types';
import {
  defaultExperimentName,
  isDefaultExperiment,
} from '@studio/components/evaluation/defaultExperiment';
import { nameFieldSlots } from '@studio/components/form/entityNameCheck';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import type { FC } from 'react';
import { Controller, useFormContext } from 'react-hook-form';

/** Radix Select reserves the empty string, so "no experiment" needs a value of its own. */
const NO_EXPERIMENT = '__none__';

interface RunTargetFieldsProps {
  workspace: string;
  agent: string;
  disabled?: boolean;
}

/** Optional experiment and evaluation name. Both can be left alone: the run then publishes under
 *  a generated name in the agent's default experiment, and can be linked to a real one later. */
export const RunTargetFields: FC<RunTargetFieldsProps> = ({ workspace, agent, disabled }) => {
  const { control, formState } = useFormContext<RunTargetFormFields>();
  const { preview, status } = useRunNameCheck(workspace, control);
  const { data: page, isLoading } = useListExperiments(workspace, {
    page_size: DEFAULT_LARGE_PAGE_SIZE,
  });
  const experiments = (page?.data ?? []).filter((experiment) => !isDefaultExperiment(experiment));
  const noneLabel = `None — link later (kept in ${defaultExperimentName(agent)})`;

  return (
    <>
      <Controller
        control={control}
        name="experimentId"
        render={({ field }) => (
          <FormField
            slotLabel="Experiment (Optional)"
            slotHelp="Groups this run with others for comparison. You can link it after the run too."
          >
            <SelectRoot
              value={field.value || NO_EXPERIMENT}
              onValueChange={(value: string) =>
                field.onChange(value === NO_EXPERIMENT ? '' : value)
              }
              disabled={disabled || isLoading}
            >
              <SelectTrigger
                className="w-full"
                aria-label="Experiment"
                placeholder={isLoading ? 'Loading experiments...' : noneLabel}
                renderValue={(value) =>
                  value === NO_EXPERIMENT
                    ? noneLabel
                    : (experiments.find((experiment) => experiment.id === value)?.name ?? undefined)
                }
              />
              <SelectContent className="w-(--radix-popper-anchor-width)">
                <SelectListbox>
                  <SelectItem value={NO_EXPERIMENT}>{noneLabel}</SelectItem>
                  {experiments.map((experiment) => (
                    <SelectItem key={experiment.id} value={experiment.id}>
                      {experiment.name}
                    </SelectItem>
                  ))}
                </SelectListbox>
              </SelectContent>
            </SelectRoot>
          </FormField>
        )}
      />
      <ControlledTextInput
        useControllerProps={{ control, name: 'evaluationRecordName' }}
        placeholder="e.g. nemotron-super-3-temp-1"
        formFieldProps={{
          slotLabel: 'Evaluation Name (Optional)',
          ...nameFieldSlots({
            entity: 'evaluation',
            preview,
            status,
            schemaError: formState.errors.evaluationRecordName?.message,
            describe: 'Results publish under this name. Generated from the agent name if blank.',
          }),
        }}
      />
    </>
  );
};
