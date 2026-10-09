// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { useInsightsGetStatusesAnalysisRunStatus } from '@nemo/sdk/generated/insights/insights-analysis-run-statuses';
import type { AnalysisConfig } from '@nemo/sdk/generated/insights/schema';
import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import {
  Button,
  Flex,
  FormField,
  Modal,
  Select,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
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
import { type FC, useState } from 'react';

const ALL_TRACES = '__all__';

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
  const { data: runStatus } = useInsightsGetStatusesAnalysisRunStatus(workspace, agent, {
    query: { retry: false, staleTime: 0 },
  });
  const periodicCursor = runStatus?.last_successful_run_at ?? undefined;
  const [preset, setPreset] = useState<SincePreset>('all');
  const [custom, setCustom] = useState(() => toDateTimeLocalValue(new Date(Date.now() - DAY_MS)));
  const [evaluation, setEvaluation] = useState(ALL_TRACES);
  const [defaultModel, setDefaultModel] = useState('');
  const [fastModel, setFastModel] = useState('');
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

  const handleRun = () =>
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
    <Modal
      open
      onOpenChange={(open) => !open && !running && onClose()}
      slotHeading={
        <Stack gap="1">
          <Text kind="title/sm">Run insight analysis</Text>
          <Text kind="body/regular/sm" className="text-secondary">
            Choose which of <strong>{agent}</strong>&apos;s traces the analyst reads for this run.
          </Text>
        </Stack>
      }
      className="w-[90vw] max-w-[640px]"
      attributes={{ ModalFooter: { className: 'justify-end' } }}
      slotFooter={
        <Flex gap="density-sm">
          <Button kind="tertiary" onClick={onClose} disabled={running}>
            Cancel
          </Button>
          <LoadingButton
            color="brand"
            onClick={handleRun}
            loading={running}
            disabled={running || invalidCustom || invalidModels}
          >
            Run insight analysis
          </LoadingButton>
        </Flex>
      }
    >
      <Stack gap="density-lg">
        <FormField
          slotLabel="Traces to analyze"
          slotHelp={
            invalidCustom
              ? 'Enter a date and time in the past.'
              : since
                ? `Analyzes traces that started since ${formatDateTime(since)}.`
                : "Analyzes the agent's full trace history."
          }
        >
          <Select
            aria-label="Traces to analyze"
            value={preset}
            onValueChange={(value) => setPreset(value as SincePreset)}
            items={presetItems}
          />
        </FormField>

        {preset === 'custom' ? (
          <FormField slotLabel="Analyze traces since">
            <TextInput
              type="datetime-local"
              value={custom}
              max={toDateTimeLocalValue(new Date())}
              onChange={(event) => setCustom(event.target.value)}
              aria-label="Analyze traces since"
            />
          </FormField>
        ) : null}

        <FormField
          slotLabel="Scope to an evaluation"
          slotHelp={
            evaluationNames.length > 0
              ? 'Reads only the spans recorded for that evaluation, within the time range above.'
              : 'No evaluations have recorded traces for this agent.'
          }
        >
          <Select
            aria-label="Scope to an evaluation"
            value={evaluation}
            onValueChange={setEvaluation}
            disabled={evaluationNames.length === 0}
            items={[
              { value: ALL_TRACES, children: 'All traces' },
              ...evaluationNames.map((name) => ({ value: name, children: name })),
            ]}
          />
        </FormField>

        <InsightsModelPairFields
          workspace={workspace}
          agent={agent}
          unresolved={false}
          defaultModel={defaultModel}
          fastModel={fastModel}
          onDefaultModelChange={setDefaultModel}
          onFastModelChange={setFastModel}
          stored={{ defaultModel: config.default_model, fastModel: config.fast_model }}
        />
      </Stack>
    </Modal>
  );
};
