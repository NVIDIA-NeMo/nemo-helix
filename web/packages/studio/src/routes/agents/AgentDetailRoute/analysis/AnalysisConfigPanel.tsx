// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getInsightsGetAnalysisConfigQueryKey,
  useInsightsGetAnalysisConfig,
} from '@nemo/sdk/generated/insights/insights-analysis-configs';
import {
  Badge,
  Button,
  Flex,
  FormField,
  Grid,
  Stack,
  Switch,
  Text,
} from '@nvidia/foundations-react-core';
import { isQualifiedModelRef } from '@studio/api/insightsAnalysis';
import { queryClient } from '@studio/api/queryClient';
import { useLatestAnalysisRun } from '@studio/api/useLatestAnalysisRun';
import { useTriggerInsightsRun } from '@studio/api/useTriggerInsightsRun';
import { LatestAnalysisRun } from '@studio/routes/agents/AgentDetailRoute/analysis/LatestAnalysisRun';
import { saveAnalysisConfig } from '@studio/routes/agents/AgentDetailRoute/analysis/saveAnalysisConfig';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import { type FC, useEffect, useState } from 'react';

interface AnalysisConfigPanelProps {
  workspace: string;
  agent?: string;
}

/**
 * The stored per-agent insights analysis config: whether the periodic controller runs, and the
 * model pair it uses.
 */
export const AnalysisConfigPanel: FC<AnalysisConfigPanelProps> = ({ workspace, agent }) => {
  const toast = useToast();
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [defaultModel, setDefaultModel] = useState('');
  const [fastModel, setFastModel] = useState('');
  const triggerRun = useTriggerInsightsRun(workspace);
  const { latestRun, isActive: runActive } = useLatestAnalysisRun(workspace, agent);

  const {
    data: config,
    isLoading,
    isError,
    error,
  } = useInsightsGetAnalysisConfig(workspace, agent ?? '', {
    query: { enabled: !!agent, retry: false },
  });

  const notFound =
    isError && (error as { response?: { status?: number } })?.response?.status === 404;

  useEffect(() => {
    setEnabled(config?.enabled ?? false);
    setDefaultModel(config?.default_model ?? '');
    setFastModel(config?.fast_model ?? '');
  }, [workspace, agent, config?.id, config?.enabled, config?.default_model, config?.fast_model]);

  useEffect(() => {
    setEditing(false);
  }, [workspace, agent]);

  const invalidRef = [defaultModel, fastModel].some(
    (ref) => ref.length > 0 && !isQualifiedModelRef(ref)
  );
  const incomplete = !defaultModel || !fastModel;

  const handleCancel = () => {
    setEnabled(config?.enabled ?? false);
    setDefaultModel(config?.default_model ?? '');
    setFastModel(config?.fast_model ?? '');
    setEditing(false);
  };

  const handleSave = async () => {
    if (!agent) return;
    setSaving(true);
    try {
      await saveAnalysisConfig(workspace, agent, { enabled, defaultModel, fastModel }, config);
      toast.success('Analysis config saved.');
      setEditing(false);
    } catch (saveError) {
      toast.error(
        saveError instanceof Error ? saveError.message : 'Failed to save analysis config.'
      );
    } finally {
      await queryClient.invalidateQueries({
        queryKey: getInsightsGetAnalysisConfigQueryKey(workspace, agent),
      });
      setSaving(false);
    }
  };

  const handleRunNow = () => {
    if (!agent) return;
    triggerRun.mutate(agent, {
      onSuccess: ({ status, jobName, message }) => {
        if (status === 'started') {
          toast.success(`Queued analysis run "${jobName}".`);
        } else {
          toast.error(message ?? 'Failed to start the analysis run.');
        }
      },
      onError: (runError) => toast.error(runError.message),
    });
  };

  if (!agent) return null;

  return (
    <DetailPanel
      title="Insights analysis"
      slotAction={
        editing ? (
          <Flex gap="density-sm">
            <Button
              kind="tertiary"
              size="small"
              className="h-auto"
              onClick={handleCancel}
              disabled={saving}
            >
              Cancel
            </Button>
            <LoadingButton
              color="brand"
              size="small"
              onClick={handleSave}
              loading={saving}
              disabled={saving || invalidRef || incomplete}
            >
              Save
            </LoadingButton>
          </Flex>
        ) : (
          <Flex gap="density-sm">
            <LoadingButton
              kind="secondary"
              color="neutral"
              size="small"
              height={28}
              onClick={handleRunNow}
              loading={triggerRun.isPending}
              disabled={!config || runActive}
            >
              Run analysis now
            </LoadingButton>
            <Button
              kind="secondary"
              size="small"
              onClick={() => setEditing(true)}
              disabled={isLoading}
            >
              Edit
            </Button>
          </Flex>
        )
      }
    >
      {isLoading ? (
        <Text className="text-secondary" kind="body/regular/sm">
          Loading analysis config...
        </Text>
      ) : isError && !notFound ? (
        <Text className="text-secondary" kind="body/regular/sm">
          Could not load the analysis config for this agent.
        </Text>
      ) : editing ? (
        <Stack gap="density-md">
          {notFound && (
            <Text className="text-secondary" kind="body/regular/xs">
              Analysis has never been enabled for this agent. Saving creates the config.
            </Text>
          )}
          <Switch
            size="small"
            name="analysis-enabled"
            checked={enabled}
            onCheckedChange={setEnabled}
            slotLabel="Run periodic analysis"
          />
          <Grid cols={{ base: 1, md: 2 }} gap="4">
            <FormField
              slotLabel="Default model"
              slotHelp="Used for quality-critical analysis work."
              slotError={
                defaultModel && !isQualifiedModelRef(defaultModel)
                  ? `Stored value "${defaultModel}" is not workspace-qualified. Pick a model to replace it.`
                  : undefined
              }
            >
              <WorkspaceModelSelect
                workspace={workspace}
                value={defaultModel ? { model: defaultModel } : null}
                onValueChange={({ model }) => setDefaultModel(model)}
                placeholder="Select a default model"
                hideAdapters
                fullWidth
                aria-label="Default model"
              />
            </FormField>
            <FormField
              slotLabel="Fast model"
              slotHelp="Used for latency-sensitive analysis work."
              slotError={
                fastModel && !isQualifiedModelRef(fastModel)
                  ? `Stored value "${fastModel}" is not workspace-qualified. Pick a model to replace it.`
                  : undefined
              }
            >
              <WorkspaceModelSelect
                workspace={workspace}
                value={fastModel ? { model: fastModel } : null}
                onValueChange={({ model }) => setFastModel(model)}
                placeholder="Select a fast model"
                hideAdapters
                fullWidth
                aria-label="Fast model"
              />
            </FormField>
          </Grid>
        </Stack>
      ) : notFound ? (
        <Text className="text-secondary" kind="body/regular/sm">
          Analysis is not enabled for this agent. Choose Edit to enable it, or run{' '}
          <code>nemo insights analysis enable --agent {agent}</code>.
        </Text>
      ) : (
        <Grid cols={{ base: 1, md: 2 }} gap="4">
          {latestRun ? (
            <div className="md:col-span-2">
              <LatestAnalysisRun workspace={workspace} run={latestRun} isActive={runActive} />
            </div>
          ) : null}
          <KVPair
            orientation="vertical"
            label="Periodic analysis"
            value={
              <Badge kind="solid" color={config?.enabled ? 'green' : 'gray'}>
                {config?.enabled ? 'Enabled' : 'Disabled'}
              </Badge>
            }
          />
          <KVPair
            orientation="vertical"
            label="Updated"
            value={config?.updated_at && <RelativeTime datetime={config.updated_at} />}
          />
          <KVPair orientation="vertical" label="Default model" value={config?.default_model} />
          <KVPair orientation="vertical" label="Fast model" value={config?.fast_model} />
        </Grid>
      )}
    </DetailPanel>
  );
};
