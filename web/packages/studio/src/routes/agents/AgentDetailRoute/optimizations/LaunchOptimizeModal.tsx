// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getAgentOptimizationListRunStrategyJobsQueryKey } from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import type { OptimizationStrategy } from '@nemo/sdk/generated/agent-optimization/schema/OptimizationStrategy';
import type { RunStrategyJob } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategyJob';
import { Button, Flex, Text } from '@nvidia/foundations-react-core';
import { useLaunchOptimizeStudy } from '@studio/api/agents/useLaunchOptimizeStudy';
import { BundleSourcePicker } from '@studio/components/BundleSourcePicker';
import { useBundleSource } from '@studio/components/BundleSourcePicker/useBundleSource';
import {
  optimizeConfigSpec,
  STRATEGY_CONFIG_SPEC,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizeConfigSpec';
import { getAgentOptimizationsTabRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useMemo } from 'react';
import { useNavigate } from 'react-router';

interface LaunchOptimizeModalProps {
  open: boolean;
  onClose: () => void;
  workspace: string;
  agentName: string;
  /**
   * An installed strategy, from the platform's listing, to run instead of Studio's own optimize
   * bundle flow. Its config is checked only for being YAML, and is optional unless the strategy
   * says it requires one.
   */
  strategy?: OptimizationStrategy;
}

const bundleTitle = (strategy: OptimizationStrategy | undefined): string => {
  if (!strategy) return 'Optimize bundle';
  if (!strategy.requires_config) return 'Config bundle (optional)';
  return 'Config bundle';
};

/**
 * Starts a study from a config the user brings. Without a `strategy` this is the optimize-bundle
 * upload, preflighted as `config-tuning` would; with one it runs that strategy, titled and
 * described in its own words.
 */
export const LaunchOptimizeModal: FC<LaunchOptimizeModalProps> = ({
  open,
  onClose,
  workspace,
  agentName,
  strategy,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const spec = useMemo(
    () => (strategy ? STRATEGY_CONFIG_SPEC : optimizeConfigSpec(agentName)),
    [agentName, strategy]
  );
  const bundle = useBundleSource({ workspace, spec });
  const { selection } = bundle.active;
  const configOptional = !!strategy && !strategy.requires_config;
  const hasPicked =
    bundle.active.isLoading ||
    !!bundle.active.error ||
    (bundle.source === 'upload' ? !!bundle.upload.summary : !!bundle.fileset.fileset);
  const canSubmit = configOptional ? !hasPicked || !!selection : !!selection;

  const {
    mutate: launch,
    error: launchError,
    isPending,
    reset: resetLaunch,
  } = useLaunchOptimizeStudy({
    onSuccess: (job: RunStrategyJob) => {
      toast.success(`Optimization study "${job.name}" submitted`);
      void queryClient.invalidateQueries({
        queryKey: getAgentOptimizationListRunStrategyJobsQueryKey(workspace),
      });
      onClose();
      navigate(getAgentOptimizationsTabRoute(workspace, agentName));
    },
  });

  const close = () => {
    resetLaunch();
    bundle.reset();
    onClose();
  };

  const errorText = launchError
    ? getErrorMessage(launchError) || 'Failed to submit the study'
    : undefined;

  const submit = () => {
    if (selection) {
      launch({
        workspace,
        agentName,
        ...selection.source,
        optimizeConfig: selection.path,
        ...(strategy ? { strategy: strategy.name } : {}),
      });
    } else if (strategy && configOptional) {
      launch({ workspace, agentName, strategy: strategy.name });
    }
  };

  return (
    <FormModal
      open={open}
      onClose={close}
      className="w-[720px] max-w-[90vw]"
      title={strategy?.title ?? 'Optimize agent'}
      instruction={
        strategy
          ? strategy.description || undefined
          : "Choose an optimize bundle — the optimize config plus the dataset and any other files it references — from your computer or a fileset. Each trial runs this agent with one set of parameters from the config's search space."
      }
      submitButtonText="Start study"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      disabled={isPending}
      loading={isPending}
      submitDisabled={!canSubmit}
      errorText={errorText}
    >
      <BundleSourcePicker
        workspace={workspace}
        state={bundle}
        title={bundleTitle(strategy)}
        filesetInfo="The fileset is the bundle root. The study runs from it in place, and deleting the study keeps it."
        disabled={isPending}
      />
      {configOptional ? (
        <Flex align="center" justify="between" gap="density-md">
          <Text kind="body/regular/sm" color="secondary">
            {hasPicked
              ? 'The study runs with this config.'
              : 'No config picked — the study runs with the strategy’s defaults.'}
          </Text>
          {hasPicked ? (
            <Button
              type="button"
              kind="tertiary"
              size="small"
              onClick={bundle.reset}
              disabled={isPending}
            >
              Run without a config
            </Button>
          ) : null}
        </Flex>
      ) : null}
    </FormModal>
  );
};
