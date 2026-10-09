// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import type { AnalysisConfig } from '@nemo/sdk/generated/insights/schema';
import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import {
  Button,
  Checkbox,
  Flex,
  FormField,
  Modal,
  Select,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { useDatasetFileContent } from '@studio/api/datasets/useDatasetFileContent';
import { isQualifiedModelRef } from '@studio/api/insightsAnalysis';
import { useLastCompletedAnalysisRun } from '@studio/api/useLastCompletedAnalysisRun';
import type { TriggerInsightsRunVariables } from '@studio/api/useTriggerInsightsRun';
import { InsightsModelPairFields } from '@studio/components/ImportTracesModal/InsightsModelPairFields';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import {
  DAY_MS,
  type SincePreset,
  sinceFor,
  toDateTimeLocalValue,
} from '@studio/routes/agents/AgentDetailRoute/analysis/analysisSince';
import { AGENT_ETHOS_FILE } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import { agentSpecFilesetName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
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
  const { data: lastCompleted, isPending: lastRunPending } = useLastCompletedAnalysisRun(
    workspace,
    agent
  );
  const lastRunAt = lastCompleted ?? undefined;
  const [chosenPreset, setPreset] = useState<SincePreset>();
  const preset = chosenPreset ?? (lastRunAt ? 'last-run' : 'day');
  const [custom, setCustom] = useState(() => toDateTimeLocalValue(new Date(Date.now() - DAY_MS)));
  const [evaluation, setEvaluation] = useState(ALL_TRACES);
  const [defaultModel, setDefaultModel] = useState(config.default_model ?? '');
  const [fastModel, setFastModel] = useState(config.fast_model ?? '');
  const [includeEthos, setIncludeEthos] = useState(true);

  const ethosFileset = agentSpecFilesetName(agent);
  const { data: ethos, isLoading: ethosLoading } = useDatasetFileContent({
    workspace,
    name: ethosFileset,
    path: AGENT_ETHOS_FILE,
    fullContent: true,
    retry: false,
  });
  const hasEthos = !!ethos?.trim();

  const { data: evaluations } = useListEvaluations(workspace, {
    page_size: DEFAULT_LARGE_PAGE_SIZE,
    sort: '-created_at',
    filter: { agent_name: agent },
  });
  const evaluationNames = (evaluations?.data ?? []).map(({ name }) => name);

  const since = sinceFor(preset, { now: new Date(), lastRunAt, custom });
  const invalidCustom = preset === 'custom' && !since;
  const invalidModels = [defaultModel, fastModel].some((ref) => !isQualifiedModelRef(ref));

  const presetItems = [
    ...(lastRunAt
      ? [
          {
            value: 'last-run',
            children: `Since the last analysis run (${formatDateTime(lastRunAt)})`,
          },
        ]
      : []),
    { value: 'day', children: 'Last 24 hours' },
    { value: 'week', children: 'Last 7 days' },
    { value: 'all', children: 'All history' },
    { value: 'custom', children: 'Custom' },
  ];

  const handleRun = () =>
    onRun({
      agent,
      overrides: { default_model: defaultModel, fast_model: fastModel },
      options: {
        since,
        evaluation_id: evaluation === ALL_TRACES ? undefined : evaluation,
        includeEthos: hasEthos && includeEthos,
      },
    });

  return (
    <Modal
      open
      onOpenChange={(open) => !open && !running && onClose()}
      slotHeading={
        <Stack gap="1">
          <Text kind="title/sm">Run analysis</Text>
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
            disabled={running || lastRunPending || invalidCustom || invalidModels}
          >
            Run analysis
          </LoadingButton>
        </Flex>
      }
    >
      <Stack gap="density-lg">
        <FormField
          slotLabel="Traces to analyze"
          slotHelp={
            lastRunPending
              ? 'Looking for the last completed analysis run...'
              : invalidCustom
                ? 'Enter a date and time.'
                : since
                  ? `Analyzes traces since ${formatDateTime(since)}.`
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
        />

        <Stack gap="density-xs">
          <Checkbox
            checked={hasEthos && includeEthos}
            disabled={!hasEthos}
            onChange={(event) => setIncludeEthos(event.target.checked)}
            slotLabel="Include the agent's Ethos"
          />
          <Text className="text-secondary" kind="body/regular/xs">
            {ethosLoading
              ? `Looking for ${AGENT_ETHOS_FILE}...`
              : hasEthos
                ? `From ${ethosFileset}/${AGENT_ETHOS_FILE}.`
                : `No ${AGENT_ETHOS_FILE} found in ${ethosFileset}.`}
          </Text>
        </Stack>
      </Stack>
    </Modal>
  );
};
