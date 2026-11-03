// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getCompassGetAnalysisConfigQueryKey,
  useCompassGetAnalysisConfig,
} from '@nemo/sdk/generated/compass/compass-analysis-configs';
import { Banner, Stack, Switch, Text } from '@nvidia/foundations-react-core';
import { isQualifiedModelRef } from '@studio/api/insightsAnalysis';
import { InsightsModelField } from '@studio/components/GetInsightsModal/InsightsModelField';
import { saveAnalysisConfig } from '@studio/routes/agents/AgentDetailRoute/analysis/saveAnalysisConfig';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useState } from 'react';

export interface InsightsSettingsModalProps {
  workspace: string;
  agent: string;
  onClose: () => void;
}

const statusOf = (error: unknown): number | undefined =>
  (error as { response?: { status?: number } } | null)?.response?.status;

/** Edits an agent's stored insights settings: the model pair and periodic analysis opt-in. */
export const InsightsSettingsModal: FC<InsightsSettingsModalProps> = ({
  workspace,
  agent,
  onClose,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [saving, setSaving] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [defaultModel, setDefaultModel] = useState('');
  const [fastModel, setFastModel] = useState('');

  const {
    data: config,
    isLoading,
    error,
  } = useCompassGetAnalysisConfig(workspace, agent, { query: { retry: false } });
  const notFound = statusOf(error) === 404;
  const unreadable = !!error && !notFound;

  useEffect(() => {
    setEnabled(config?.enabled ?? false);
    setDefaultModel(config?.default_model ?? '');
    setFastModel(config?.fast_model ?? '');
  }, [config?.id, config?.enabled, config?.default_model, config?.fast_model]);

  const invalid = [defaultModel, fastModel].some((ref) => !isQualifiedModelRef(ref));

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveAnalysisConfig(workspace, agent, { enabled, defaultModel, fastModel }, config);
      toast.success('Insights settings saved.');
      onClose();
    } catch (saveError) {
      toast.error(
        saveError instanceof Error ? saveError.message : 'Failed to save insights settings.'
      );
    } finally {
      await queryClient.invalidateQueries({
        queryKey: getCompassGetAnalysisConfigQueryKey(workspace, agent),
      });
      setSaving(false);
    }
  };

  return (
    <FormModal
      open
      title={
        <Stack gap="1">
          <Text kind="title/sm">Insights settings</Text>
          <Text kind="body/regular/sm" className="text-secondary">
            Models and scheduling for <strong>{agent}</strong>&apos;s insights analysis.
          </Text>
        </Stack>
      }
      submitButtonText="Save"
      className="w-[90vw] max-w-[560px]"
      onSubmit={(event) => {
        event.preventDefault();
        void handleSave();
      }}
      onClose={onClose}
      disabled={saving}
      loading={saving}
      submitDisabled={isLoading || unreadable || invalid}
      attributes={{ SubmitButton: { color: 'brand' } }}
    >
      <Stack gap="density-lg">
        {unreadable ? (
          <Banner kind="inline" status="error">
            Could not load the insights settings for this agent.
          </Banner>
        ) : notFound ? (
          <Banner kind="inline" status="info">
            Insights aren&apos;t set up for this agent yet. Pick both models to get started.
          </Banner>
        ) : null}
        <InsightsModelField
          workspace={workspace}
          label="Default model"
          help="Used for quality-critical analysis work."
          value={defaultModel}
          onChange={setDefaultModel}
          disabled={isLoading || unreadable}
        />
        <InsightsModelField
          workspace={workspace}
          label="Fast model"
          help="Used for latency-sensitive analysis work."
          value={fastModel}
          onChange={setFastModel}
          disabled={isLoading || unreadable}
        />
        <Switch
          size="small"
          name="analysis-enabled"
          checked={enabled}
          onCheckedChange={setEnabled}
          disabled={isLoading || unreadable}
          slotLabel="Run periodic analysis"
        />
        <Text kind="body/regular/xs" className="text-secondary -mt-2">
          Analyzes this agent&apos;s new traces on the platform&apos;s schedule.
        </Text>
      </Stack>
    </FormModal>
  );
};
