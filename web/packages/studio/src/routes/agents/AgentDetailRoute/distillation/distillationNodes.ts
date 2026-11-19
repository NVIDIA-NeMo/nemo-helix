// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DistillationWorkflow,
  NodeStatus,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';

/** The node-key prefix each stage covers; the plugin groups `train[lora]` and `eval_distilled_x[lora]` the same way. */
const STAGE_NODE_PREFIX: Record<string, string> = {
  prepare: 'prepare',
  customization: 'train',
  'eval-original': 'eval_original',
  'eval-base': 'eval_base',
  'eval-distilled': 'eval_distilled',
};

export interface StageNode {
  key: string;
  kind: string;
  status: NodeStatus;
  attempt: number;
  job: string | undefined;
  error: string | null;
}

export const nodesForStage = (workflow: DistillationWorkflow, stageId: string): StageNode[] => {
  const prefix = STAGE_NODE_PREFIX[stageId];
  if (!prefix) return [];
  return Object.keys(workflow.graph)
    .filter((key) => {
      const head = key.split('[', 1)[0];
      return head === prefix || head.startsWith(`${prefix}_`);
    })
    .flatMap((key) => {
      const runtime = workflow.nodes[key];
      if (!runtime) return [];
      return [
        {
          key,
          kind: workflow.graph[key].kind,
          status: runtime.status,
          attempt: runtime.attempt,
          job: runtime.external.job,
          error: runtime.error,
        },
      ];
    });
};
