// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';
import {
  getCompassGetAnalysisConfigQueryKey,
  useCompassGetAnalysisConfig,
} from '@nemo/sdk/generated/compass/compass-analysis-configs';
import { useCompassGetStatusesAnalysisRunStatus } from '@nemo/sdk/generated/compass/compass-analysis-run-statuses';
import { Banner, DatePicker, FormField, Grid, Stack, Text } from '@nvidia/foundations-react-core';
import { isQualifiedModelRef } from '@studio/api/insightsAnalysis';
import { useLatestAnalysisRun } from '@studio/api/useLatestAnalysisRun';
import { useTriggerInsightsRun } from '@studio/api/useTriggerInsightsRun';
import { InsightsModelField } from '@studio/components/GetInsightsModal/InsightsModelField';
import {
  DAY_MS,
  type SincePreset,
  sinceFor,
} from '@studio/routes/agents/AgentDetailRoute/analysis/analysisSince';
import { saveAnalysisConfig } from '@studio/routes/agents/AgentDetailRoute/analysis/saveAnalysisConfig';
import {
  InsightsView,
  TAB_SEARCH_PARAM,
  VIEW_SEARCH_PARAM,
} from '@studio/routes/agents/AgentDetailRoute/tabs';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { formatDateTime } from '@studio/util/date';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useState } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';

interface GetInsightsFormFields {
  agent: string;
  preset: SincePreset;
  /** ISO start of the picked day; empty until one is picked. */
  custom: string;
  /** Only asked for while the agent has no insights settings yet. */
  defaultModel: string;
  fastModel: string;
}

/** The agents list API caps page_size at 100. */
const AGENTS_PAGE_SIZE = 100;

const startOfDay = (date: Date) => new Date(new Date(date).setHours(0, 0, 0, 0));

export interface GetInsightsModalProps {
  workspace: string;
  /** Preselects and locks the agent, as on an agent's own page. */
  agent?: string;
  onClose: () => void;
}

const statusOf = (error: unknown): number | undefined =>
  (error as { response?: { status?: number } } | null)?.response?.status;

/**
 * Starts an ad-hoc insights analysis run for one agent, then opens the agent's Insights jobs.
 *
 * The default model comes from the agent's stored insights settings, and the fast model can be
 * replaced for this run. An agent with no settings yet picks both here, and they are saved to its
 * settings (periodic analysis stays off) before the run starts.
 */
export const GetInsightsModal: FC<GetInsightsModalProps> = ({
  workspace,
  agent: lockedAgent,
  onClose,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const triggerRun = useTriggerInsightsRun(workspace);
  const [savingSetup, setSavingSetup] = useState(false);

  const { control, handleSubmit, setValue } = useForm<GetInsightsFormFields>({
    defaultValues: {
      agent: lockedAgent ?? '',
      preset: 'two-weeks',
      custom: startOfDay(new Date(Date.now() - DAY_MS)).toISOString(),
      defaultModel: '',
      fastModel: '',
    },
  });
  const [pickedAgent, preset, custom, setupDefaultModel, fastModel] = useWatch({
    control,
    name: ['agent', 'preset', 'custom', 'defaultModel', 'fastModel'],
  });
  const agent = lockedAgent ?? pickedAgent;

  const { data: agentsResponse, isLoading: isAgentsLoading } = useAgentsListAgents(
    workspace,
    { page_size: AGENTS_PAGE_SIZE },
    { query: { enabled: !lockedAgent } }
  );
  const agentNames = (agentsResponse?.data ?? [])
    .map(({ name }) => name)
    .filter((name): name is string => !!name);

  const {
    data: config,
    isLoading: isConfigLoading,
    error: configError,
  } = useCompassGetAnalysisConfig(workspace, agent, {
    query: { enabled: !!agent, retry: false },
  });

  const {
    data: runStatus,
    isFetchedAfterMount,
    isError: isRunStatusError,
  } = useCompassGetStatusesAnalysisRunStatus(workspace, agent, {
    query: { enabled: !!agent, retry: false, staleTime: 0 },
  });
  const periodicCursor =
    isFetchedAfterMount && !isRunStatusError
      ? (runStatus?.last_successful_run_at ?? undefined)
      : undefined;

  const { blocksNewRun } = useLatestAnalysisRun(workspace, agent || undefined, { notify: false });

  const needsSetup =
    !!agent &&
    !isConfigLoading &&
    (statusOf(configError) === 404 || (!!config && !config.default_model));
  const configUnreadable = !!agent && !!configError && statusOf(configError) !== 404;
  const defaultModel = needsSetup ? setupDefaultModel : (config?.default_model ?? '');
  const runFastModel = fastModel || config?.fast_model || '';

  const since = sinceFor(preset, { now: new Date(), periodicCursor, custom });
  const invalidCustom = preset === 'custom' && !since;
  const missingCursor = preset === 'periodic-cursor' && !periodicCursor;
  const invalidModels = [defaultModel, runFastModel].some((ref) => !isQualifiedModelRef(ref));
  const busy = savingSetup || triggerRun.isPending;

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
    { value: 'two-weeks', children: 'Last two weeks' },
    { value: 'month', children: 'Last month' },
    { value: 'custom', children: 'Custom start date' },
  ];

  /** Saves the picked pair as the agent's settings, keeping its periodic opt-in as it was. */
  const saveSetup = async () => {
    setSavingSetup(true);
    try {
      return await saveAnalysisConfig(
        workspace,
        agent,
        { enabled: config?.enabled ?? false, defaultModel, fastModel: runFastModel },
        config
      );
    } finally {
      await queryClient.invalidateQueries({
        queryKey: getCompassGetAnalysisConfigQueryKey(workspace, agent),
      });
      setSavingSetup(false);
    }
  };

  const onSubmit = async () => {
    let storedConfig = config;
    if (needsSetup) {
      try {
        storedConfig = await saveSetup();
      } catch (saveError) {
        toast.error(
          saveError instanceof Error ? saveError.message : 'Failed to save insights settings.'
        );
        return;
      }
    }
    if (!storedConfig) return;
    triggerRun.mutate(
      {
        agent,
        overrides: { default_model: defaultModel, fast_model: runFastModel },
        options: { since, storedConfig },
      },
      {
        onSuccess: ({ status, jobName, message }) => {
          if (status === 'started' && jobName) {
            toast.success(`Queued analysis run "${jobName}".`);
            onClose();
            navigate({
              pathname: getAgentDetailRoute(workspace, agent),
              search: new URLSearchParams({
                [TAB_SEARCH_PARAM]: 'insights',
                [VIEW_SEARCH_PARAM]: InsightsView.Jobs,
              }).toString(),
            });
          } else {
            toast.error(message ?? 'Failed to start the analysis run.');
          }
        },
        onError: (runError) => toast.error(runError.message),
      }
    );
  };

  const onAgentChange = () => {
    setValue('defaultModel', '');
    setValue('fastModel', '');
    if (preset === 'periodic-cursor') setValue('preset', 'two-weeks');
  };

  return (
    <FormModal
      open
      title={
        <Stack gap="1">
          <Text kind="title/sm">Get Insights</Text>
          <Text kind="body/regular/sm" className="text-secondary">
            {lockedAgent ? (
              <>
                Analyze <strong>{lockedAgent}</strong>&apos;s traces to find recurring failure
                patterns.
              </>
            ) : (
              "Analyze an agent's traces to find recurring failure patterns."
            )}
          </Text>
        </Stack>
      }
      submitButtonText="Get Insights"
      className="w-[90vw] max-w-[640px]"
      onSubmit={(event) => void handleSubmit(onSubmit)(event)}
      onClose={onClose}
      disabled={busy}
      loading={busy}
      submitDisabled={
        !agent ||
        isConfigLoading ||
        configUnreadable ||
        blocksNewRun ||
        invalidCustom ||
        missingCursor ||
        invalidModels
      }
      attributes={{ SubmitButton: { color: 'brand' } }}
    >
      <Stack gap="density-lg">
        {lockedAgent ? null : (
          <ControlledSelect
            useControllerProps={{ control, name: 'agent' }}
            aria-label="Agent"
            placeholder="Select an agent"
            loading={isAgentsLoading}
            onChange={onAgentChange}
            items={agentNames.map((name) => ({
              value: name,
              children: name,
            }))}
            formFieldProps={{
              slotLabel: 'Agent',
              slotHelp:
                !isAgentsLoading && agentNames.length === 0
                  ? 'No agents are registered in this workspace.'
                  : undefined,
            }}
          />
        )}

        {needsSetup ? (
          <Banner kind="inline" status="info">
            Pick the models Compass uses for <strong>{agent}</strong>. They&apos;re saved to the
            agent&apos;s insights settings.
          </Banner>
        ) : configUnreadable ? (
          <Banner kind="inline" status="error">
            Could not load the insights settings for <strong>{agent}</strong>.
          </Banner>
        ) : blocksNewRun ? (
          <Banner kind="inline" status="info">
            An analysis run for <strong>{agent}</strong> is still in progress. You can start another
            run when it finishes.
          </Banner>
        ) : null}

        {needsSetup ? (
          <InsightsModelField
            workspace={workspace}
            label="Default model"
            help="Used for quality-critical analysis work."
            value={setupDefaultModel}
            onChange={(value) => setValue('defaultModel', value)}
          />
        ) : null}
        <InsightsModelField
          workspace={workspace}
          label="Fast model"
          help={needsSetup ? 'Used for latency-sensitive analysis work.' : undefined}
          value={runFastModel}
          onChange={(value) => setValue('fastModel', value)}
          disabled={!agent || isConfigLoading || configUnreadable}
        />

        <Grid cols={2} gap="4" className="items-start">
          <ControlledSelect
            useControllerProps={{ control, name: 'preset' }}
            aria-label="Traces since"
            items={presetItems}
            formFieldProps={{
              slotLabel: 'Traces since',
              slotHelp: invalidCustom
                ? custom
                  ? 'Pick a start date that is not in the future.'
                  : 'Pick a start date.'
                : missingCursor
                  ? 'The last periodic analysis time could not be read. Pick another range.'
                  : since
                    ? `Analyzes traces that started since ${formatDateTime(since)}.`
                    : "Analyzes the agent's full trace history.",
            }}
          />

          {preset === 'custom' ? (
            <FormField slotLabel="Start date">
              <DatePicker
                format="yyyy-MM-dd"
                placeholder="yyyy-mm-dd"
                value={custom ? new Date(custom) : undefined}
                onValueChange={(date) =>
                  setValue('custom', date ? startOfDay(date).toISOString() : '')
                }
                disabledMatcher={{ after: new Date() }}
              />
            </FormField>
          ) : null}
        </Grid>
      </Stack>
    </FormModal>
  );
};
