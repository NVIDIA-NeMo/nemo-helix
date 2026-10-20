// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { OptimizationStrategy } from '@nemo/sdk/generated/agent-optimization/schema/OptimizationStrategy';
import { STUDIO_OPTIMIZE_STRATEGY } from '@studio/api/agents/useLaunchOptimizeStudy';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { NewOptimizationForm } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm';
import { useSubmitOptimization } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/useSubmitOptimization';
import { OptimizationStrategySelect } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect';
import { SWITCHYARD_STRATEGY } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/constants';
import { OptimizeJobsTable } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizeJobsTable';
import { SwitchyardOptimizationForm } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm';
import { useSubmitSwitchyardOptimization } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/useSubmitSwitchyardOptimization';
import { OptimizationView } from '@studio/routes/agents/AgentDetailRoute/tabs';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { type FC } from 'react';

export interface OptimizationsTabProps {
  agentName?: string;
  evals: AgentEvaluationRow[];
  isEvalsPending: boolean;
  view: OptimizationView;
  onOptimize?: () => void;
  onViewChange: (view: OptimizationView) => void;
  /** Opens the launch modal for uploading a `config-tuning` bundle. */
  onUploadConfig: () => void;
  /** Opens the launch modal for an installed strategy Studio has no form of its own for. */
  onRunStrategy: (strategy: OptimizationStrategy) => void;
}

/** The tab's views: the studies table, the strategy picker, and the forms that create one. The
 *  picker and forms take over the tab rather than opening a dialog, so each has its own back button;
 *  only the upload tile and installed strategies without a form of their own hand off to a
 *  dialog, the launch modal the route owns. */
export const OptimizationsTab: FC<OptimizationsTabProps> = ({
  agentName,
  evals,
  isEvalsPending,
  view,
  onOptimize,
  onViewChange,
  onUploadConfig,
  onRunStrategy,
}) => {
  const workspace = useWorkspaceFromPath();
  const submitOptimization = useSubmitOptimization({ workspace, agentName, evals });
  const submitRouting = useSubmitSwitchyardOptimization({ workspace, agentName });
  const forms: Record<string, OptimizationView> = {
    [STUDIO_OPTIMIZE_STRATEGY]: OptimizationView.Form,
    [SWITCHYARD_STRATEGY]: OptimizationView.Routing,
  };

  switch (view) {
    case OptimizationView.Strategy:
      return (
        <OptimizationStrategySelect
          agentName={agentName}
          onBack={() => onViewChange(OptimizationView.Table)}
          onSelect={(strategy) => {
            const form = forms[strategy.name];
            if (form) onViewChange(form);
            else onRunStrategy(strategy);
          }}
          onUploadConfig={onUploadConfig}
        />
      );
    case OptimizationView.Form:
      return (
        <NewOptimizationForm
          key={agentName}
          agentName={agentName}
          evals={evals}
          isEvalsPending={isEvalsPending}
          onBack={() => onViewChange(OptimizationView.Strategy)}
          onSubmit={submitOptimization}
        />
      );
    case OptimizationView.Routing:
      return (
        <SwitchyardOptimizationForm
          key={agentName}
          agentName={agentName}
          onBack={() => onViewChange(OptimizationView.Strategy)}
          onSubmit={submitRouting}
        />
      );
    case OptimizationView.Table:
      return <OptimizeJobsTable agentName={agentName} onOptimize={onOptimize} />;
  }
};
