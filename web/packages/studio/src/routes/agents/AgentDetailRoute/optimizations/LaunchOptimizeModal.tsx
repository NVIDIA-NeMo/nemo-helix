// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getAgentOptimizationListRunStrategyJobsQueryKey } from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import type { RunStrategyJob } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategyJob';
import { useLaunchOptimizeStudy } from '@studio/api/agents/useLaunchOptimizeStudy';
import { BundleSourcePicker } from '@studio/components/BundleSourcePicker';
import { useBundleSource } from '@studio/components/BundleSourcePicker/useBundleSource';
import { optimizeConfigSpec } from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizeConfigSpec';
import { getAgentOptimizationsTabRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useMemo } from 'react';
import { useNavigate } from 'react-router';

interface LaunchOptimizeModalProps {
  open: boolean;
  onClose: () => void;
  workspace: string;
  agentName: string;
}

export const LaunchOptimizeModal: FC<LaunchOptimizeModalProps> = ({
  open,
  onClose,
  workspace,
  agentName,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const spec = useMemo(() => optimizeConfigSpec(agentName), [agentName]);
  const bundle = useBundleSource({ workspace, spec });
  const { selection } = bundle.active;

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

  return (
    <FormModal
      open={open}
      onClose={close}
      className="w-[720px] max-w-[90vw]"
      title="Optimize agent"
      instruction="Choose an optimize bundle — the optimize config plus the dataset and any other files it references — from your computer or a fileset. Each trial runs this agent with one set of parameters from the config's search space."
      submitButtonText="Start study"
      onSubmit={(event) => {
        event.preventDefault();
        if (!selection) return;
        launch({ workspace, agentName, ...selection.source, optimizeConfig: selection.path });
      }}
      disabled={isPending}
      loading={isPending}
      submitDisabled={!selection}
      errorText={errorText}
    >
      <BundleSourcePicker
        workspace={workspace}
        state={bundle}
        title="Optimize bundle"
        filesetInfo="The fileset is the bundle root. The study runs from it in place, and deleting the study keeps it."
        disabled={isPending}
      />
    </FormModal>
  );
};
