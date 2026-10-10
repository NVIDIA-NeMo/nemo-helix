// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { NewOptimizationForm } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm';
import { useSubmitOptimization } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/useSubmitOptimization';
import { OptimizationStrategySelect } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect';
import type { OptimizationStrategyId } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/types';
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
  onUploadConfig: () => void;
}

/** The tab's views: the studies table, the strategy picker, and the forms that create one. The
 *  picker and forms take over the tab rather than opening a dialog, so each has its own back button;
 *  only the upload strategy hands off to a dialog, the launch modal the route owns. */
export const OptimizationsTab: FC<OptimizationsTabProps> = ({
  agentName,
  evals,
  isEvalsPending,
  view,
  onOptimize,
  onViewChange,
  onUploadConfig,
}) => {
  const workspace = useWorkspaceFromPath();
  const submitOptimization = useSubmitOptimization({ workspace, agentName, evals });
  const submitRouting = useSubmitSwitchyardOptimization({ workspace, agentName });
  // A record rather than a switch, so a strategy added without a handler fails to compile.
  const continueWith: Record<OptimizationStrategyId, () => void> = {
    hyperparameter: () => onViewChange(OptimizationView.Form),
    routing: () => onViewChange(OptimizationView.Routing),
    upload: onUploadConfig,
  };

  switch (view) {
    case OptimizationView.Strategy:
      return (
        <OptimizationStrategySelect
          agentName={agentName}
          onBack={() => onViewChange(OptimizationView.Table)}
          onSelect={(strategy) => continueWith[strategy]()}
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
