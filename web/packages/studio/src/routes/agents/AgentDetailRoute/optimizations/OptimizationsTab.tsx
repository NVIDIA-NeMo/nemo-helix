// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NewOptimizationForm } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm';
import { OptimizeJobsTable } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizeJobsTable';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { type FC } from 'react';

export interface OptimizationsTabProps {
  agentName?: string;
  evals: AgentEvaluationRow[];
  isEvalsPending: boolean;
  isCreating: boolean;
  onCloseForm: () => void;
}

/** The tab's two views: the studies table, and the form that creates one. The form takes over the
 *  tab rather than opening a dialog, so its own breadcrumb is the way back. */
export const OptimizationsTab: FC<OptimizationsTabProps> = ({
  agentName,
  evals,
  isEvalsPending,
  isCreating,
  onCloseForm,
}) =>
  isCreating ? (
    <NewOptimizationForm
      agentName={agentName}
      evals={evals}
      isEvalsPending={isEvalsPending}
      onBack={onCloseForm}
    />
  ) : (
    <OptimizeJobsTable agentName={agentName} />
  );
