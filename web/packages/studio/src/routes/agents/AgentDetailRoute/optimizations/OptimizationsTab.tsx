// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { NewOptimizationForm } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm';
import { useSubmitOptimization } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/useSubmitOptimization';
import { OptimizationStrategySelect } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect';
import type { OptimizationStrategyId } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/types';
import { OptimizeJobsTable } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizeJobsTable';
import type { OptimizationView } from '@studio/routes/agents/AgentDetailRoute/tabs';
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

/** The tab's views: the studies table, the strategy picker, and the form that creates one. The
 *  picker and form take over the tab rather than opening a dialog, so each has its own back button;
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
  // A record rather than a switch, so a strategy added without a handler fails to compile.
  const continueWith: Record<OptimizationStrategyId, () => void> = {
    form: () => onViewChange('form'),
    upload: onUploadConfig,
  };

  switch (view) {
    case 'strategy':
      return (
        <OptimizationStrategySelect
          agentName={agentName}
          onBack={() => onViewChange('table')}
          onContinue={(strategy) => continueWith[strategy]()}
        />
      );
    case 'form':
      return (
        <NewOptimizationForm
          key={agentName}
          agentName={agentName}
          evals={evals}
          isEvalsPending={isEvalsPending}
          onBack={() => onViewChange('strategy')}
          onSubmit={submitOptimization}
        />
      );
    case 'table':
      return <OptimizeJobsTable agentName={agentName} onOptimize={onOptimize} />;
  }
};
