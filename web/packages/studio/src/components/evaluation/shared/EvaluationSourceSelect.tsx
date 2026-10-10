// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSearchableSelect } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import { Text } from '@nvidia/foundations-react-core';
import type { UseEvaluationSourcesResult } from '@studio/components/evaluation/shared/useEvaluationSources';
import type { FieldPath, FieldValues } from 'react-hook-form';

interface EvaluationSourceSelectProps<T extends FieldValues> extends Pick<
  UseEvaluationSourcesResult,
  'options' | 'groupLabels' | 'byName' | 'isLoading'
> {
  name: FieldPath<T>;
  /** Name of the evaluation currently selected, used to name its experiment beneath the field. */
  selectedName: string;
  slotError?: string;
  disabled?: boolean;
}

/**
 * One picker for "which evaluation am I re-running": every reusable evaluation in the workspace,
 * sectioned by the experiment it belongs to, with a typeahead that matches an experiment's name
 * as well as an evaluation's.
 *
 * That combination is deliberate. Picking an experiment first and an evaluation second means two
 * controls, and makes the experiment a required step even when the user already knows the run by
 * name. Here typing "baseline" surfaces the baseline run of every experiment that has one, typing
 * an experiment's name narrows to that one section, and the section headings mean the chosen run's
 * experiment is never a mystery.
 */
export const EvaluationSourceSelect = <T extends FieldValues>({
  name,
  options,
  groupLabels,
  byName,
  isLoading,
  selectedName,
  slotError,
  disabled,
}: EvaluationSourceSelectProps<T>) => {
  const selected = byName[selectedName];
  const experimentName = selected?.experimentName;
  const otherAgentNames = selected?.fromOtherAgent
    ? (selected.evaluation.agent_names ?? []).join(', ')
    : '';

  return (
    <ControlledSearchableSelect
      useControllerProps={{ name }}
      options={options}
      groupLabels={groupLabels}
      isLoading={isLoading}
      disabled={disabled}
      triggerPlaceholder="Select an evaluation"
      searchPlaceholder="Search experiments and evaluations..."
      emptyMessage="No evaluation matches that search"
      formFieldProps={{
        slotLabel: 'Evaluation to Re-run',
        slotHelp: experimentName ? (
          <>
            Reuses the eval config saved on this run. Experiment:{' '}
            <Text kind="body/semibold/sm" className="text-primary">
              {experimentName}
            </Text>
            {otherAgentNames ? (
              <>
                {' '}
                · Agent:{' '}
                <Text kind="body/semibold/sm" className="text-primary">
                  {otherAgentNames}
                </Text>
              </>
            ) : null}
          </>
        ) : (
          'Search by experiment or evaluation name. The new run reuses the eval config saved on the one you pick.'
        ),
        slotError,
        status: slotError ? 'error' : undefined,
      }}
      hideError
    />
  );
};
