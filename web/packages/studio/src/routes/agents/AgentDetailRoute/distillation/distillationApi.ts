// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { customFetch } from '@nemo/sdk/generated/fetchers/platform';
import type {
  CreateWorkflowBody,
  DistillationWorkflow,
  DistillInputs,
  RenderResponse,
  WorkflowPage,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';

const base = (workspace: string) => `/apis/distillation/v2/workspaces/${workspace}`;

export const renderDistillationWorkflow = (workspace: string, inputs: DistillInputs) =>
  customFetch<RenderResponse>({
    url: `${base(workspace)}/workflow-templates/distillation:render`,
    method: 'POST',
    data: inputs,
  });

export const createDistillationWorkflow = (workspace: string, body: CreateWorkflowBody) =>
  customFetch<DistillationWorkflow>({
    url: `${base(workspace)}/workflows`,
    method: 'POST',
    data: body,
  });

export const listDistillationWorkflows = (
  workspace: string,
  filter: { source_agent?: string; status?: string } = {}
) =>
  customFetch<WorkflowPage>({
    url: `${base(workspace)}/workflows`,
    method: 'GET',
    params: { filter: { template: 'distillation', ...filter } },
  });

export const getDistillationWorkflow = (workspace: string, name: string) =>
  customFetch<DistillationWorkflow>({
    url: `${base(workspace)}/workflows/${encodeURIComponent(name)}`,
    method: 'GET',
  });

export const cancelDistillationWorkflow = (workspace: string, name: string) =>
  customFetch<DistillationWorkflow>({
    url: `${base(workspace)}/workflows/${encodeURIComponent(name)}/cancel`,
    method: 'POST',
  });

export const retryDistillationNode = (workspace: string, name: string, node: string) =>
  customFetch<DistillationWorkflow>({
    url: `${base(workspace)}/workflows/${encodeURIComponent(name)}/nodes/${encodeURIComponent(node)}/retry`,
    method: 'POST',
  });
