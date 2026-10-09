// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useInsightsGetStatusesAnalysisRunStatus } from '@nemo/sdk/generated/insights/insights-analysis-run-statuses';
import type { AnalysisConfig } from '@nemo/sdk/generated/insights/schema';
import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { isQualifiedModelRef } from '@studio/api/insightsAnalysis';
import type { TriggerInsightsRunVariables } from '@studio/api/useTriggerInsightsRun';
import { InsightsModelPairFields } from '@studio/components/ImportTracesModal/InsightsModelPairFields';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import {
  DAY_MS,
  type SincePreset,
  sinceFor,
  toDateTimeLocalValue,
} from '@studio/routes/agents/AgentDetailRoute/analysis/analysisSince';
import { formatDateTime } from '@studio/util/date';
import type { FC } from 'react';
import { useForm, useWatch } from 'react-hook-form';

const ALL_TRACES = '__all__';

interface RunAnalysisFormFields {
  preset: SincePreset;
  custom: string;
  evaluation: string;
  defaultModel: string;
  fastModel: string;
}

interface RunAnalysisModalProps {
  workspace: string;
  agent: string;
  config: AnalysisConfig;
  running: boolean;
  onClose: () => void;
  onRun: (variables: TriggerInsightsRunVariables) => void;
}

export const RunAnalysisModal: FC<RunAnalysisModalProps> = ({
  workspace,
  agent,
  config,
  running,
  onClose,
  onRun,
}) => {
  const {
    data: runStatus,
    isFetchedAfterMount,
    isError,
  } = useInsightsGetStatusesAnalysisRunStatus(workspace, agent, {
    query: { retry: false, staleTime: 0 },
  });
  const periodicCursor =
    isFetchedAfterMount && !isError ? (runStatus?.last_successful_run_at ?? undefined) : undefined;

  const { control, handleSubmit, setValue } = useForm<RunAnalysisFormFields>({
    defaultValues: {
      preset: 'all',
      custom: toDateTimeLocalValue(new Date(Date.now() - DAY_MS)),
      evaluation: ALL_TRACES,
      defaultModel: '',
      fastModel: '',
    },
  });
  const [preset, custom, defaultModel, fastModel] = useWatch({
    control,
    name: ['preset', 'custom', 'defaultModel', 'fastModel'],
  });
  const runDefaultModel = defaultModel || config.default_model || '';
  const runFastModel = fastModel || config.fast_model || '';

  // Intake fills an evaluation's agent_names up to a minute after ingest, so reread on every open.
  const { data: evaluations } = useListEvaluations(
    workspace,
    { page_size: DEFAULT_LARGE_PAGE_SIZE, sort: '-created_at', filter: { agent_name: agent } },
    { query: { staleTime: 0 } }
  );
  const evaluationNames = (evaluations?.data ?? []).map(({ name }) => name);

  const since = sinceFor(preset, { now: new Date(), periodicCursor, custom });
  const invalidCustom = preset === 'custom' && !since;
  const invalidModels = [runDefaultModel, runFastModel].some((ref) => !isQualifiedModelRef(ref));

  const presetItems = [
    { value: 'all', children: 'All history' },
    ...(periodicCursor
      ? [
          {
            value: 'periodic-cursor',
            children: `Since the last periodic analysis (${formatDateTime(periodicCursor)})`,
          },
        ]
      : []),
    { value: 'day', children: 'Last 24 hours' },
    { value: 'week', children: 'Last 7 days' },
    { value: 'custom', children: 'Custom' },
  ];

  const onSubmit = ({ evaluation }: RunAnalysisFormFields) =>
    onRun({
      agent,
      overrides: { default_model: runDefaultModel, fast_model: runFastModel },
      options: {
        since,
        evaluation_id: evaluation === ALL_TRACES ? undefined : evaluation,
        storedConfig: config,
      },
    });

  return (
    <FormModal
      open
      title={
        <Stack gap="1">
          <Text kind="title/sm">Run insight analysis</Text>
          <Text kind="body/regular/sm" className="text-secondary">
            Choose which of <strong>{agent}</strong>&apos;s traces the analyst reads for this run.
          </Text>
        </Stack>
      }
      submitButtonText="Run insight analysis"
      className="w-[90vw] max-w-[640px]"
      onSubmit={(event) => void handleSubmit(onSubmit)(event)}
      onClose={onClose}
      disabled={running}
      loading={running}
      submitDisabled={invalidCustom || invalidModels}
      attributes={{ SubmitButton: { color: 'brand' } }}
    >
      <Stack gap="density-lg">
        <ControlledSelect
          useControllerProps={{ control, name: 'preset' }}
          aria-label="Traces to analyze"
          items={presetItems}
          formFieldProps={{
            slotLabel: 'Traces to analyze',
            slotHelp: invalidCustom
              ? 'Enter a date and time in the past.'
              : since
                ? `Analyzes traces that started since ${formatDateTime(since)}.`
                : "Analyzes the agent's full trace history.",
          }}
        />

        {preset === 'custom' ? (
          <ControlledTextInput
            useControllerProps={{ control, name: 'custom' }}
            label="Analyze traces since"
            type="datetime-local"
            max={toDateTimeLocalValue(new Date())}
          />
        ) : null}

        <ControlledSelect
          useControllerProps={{ control, name: 'evaluation' }}
          aria-label="Scope to an evaluation"
          disabled={evaluationNames.length === 0}
          items={[
            { value: ALL_TRACES, children: 'All traces' },
            ...evaluationNames.map((name) => ({ value: name, children: name })),
          ]}
          formFieldProps={{
            slotLabel: 'Scope to an evaluation',
            slotHelp:
              evaluationNames.length > 0
                ? 'Reads only the spans recorded for that evaluation, within the time range above.'
                : 'No evaluations have recorded traces for this agent.',
          }}
        />

        <InsightsModelPairFields
          workspace={workspace}
          agent={agent}
          unresolved={false}
          defaultModel={defaultModel}
          fastModel={fastModel}
          onDefaultModelChange={(value) => setValue('defaultModel', value)}
          onFastModelChange={(value) => setValue('fastModel', value)}
          stored={{ defaultModel: config.default_model, fastModel: config.fast_model }}
        />
      </Stack>
    </FormModal>
  );
};
