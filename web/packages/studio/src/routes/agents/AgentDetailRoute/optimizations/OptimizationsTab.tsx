// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { NewOptimizationForm } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm';
import { useSubmitOptimization } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/useSubmitOptimization';
import { OptimizeJobsTable } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizeJobsTable';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { type FC } from 'react';

export interface OptimizationsTabProps {
  agentName?: string;
  evals: AgentEvaluationRow[];
  isEvalsPending: boolean;
  isCreating: boolean;
  onOptimize?: () => void;
  onCloseForm: () => void;
}

/** The tab's two views: the studies table, and the form that creates one. The form takes over the
 *  tab rather than opening a dialog, so its own breadcrumb is the way back. */
export const OptimizationsTab: FC<OptimizationsTabProps> = ({
  agentName,
  evals,
  isEvalsPending,
  isCreating,
  onOptimize,
  onCloseForm,
}) => {
  const workspace = useWorkspaceFromPath();
  const submitOptimization = useSubmitOptimization({ workspace, agentName, evals });

  return isCreating ? (
    <NewOptimizationForm
      key={agentName}
      agentName={agentName}
      evals={evals}
      isEvalsPending={isEvalsPending}
      onBack={onCloseForm}
      onSubmit={submitOptimization}
    />
  ) : (
    <OptimizeJobsTable agentName={agentName} onOptimize={onOptimize} />
  );
};
