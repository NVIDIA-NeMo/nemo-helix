// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsGetAgent, agentsRegisterAgent } from '@nemo/sdk/generated/agents/agents';
import type { DistillationWorkflow } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';

const FABRIC_CONFIG_FORMAT = 'nemo-agents-spec-v1';
const DEFAULT_CANDIDATE = 'lora';

const qualify = (workspace: string, model: string): string =>
  model.includes('/') ? model : `${workspace}/${model}`;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * The gateway name of the trained model, built the way the plugin names the arm it evaluates: a LoRA
 * candidate is an adapter on the student, a full fine-tune is a model of its own.
 */
export const distilledGatewayModel = (workflow: DistillationWorkflow): string | undefined => {
  const candidate = workflow.inputs.candidates?.[0];
  const trainKey =
    `train[${candidate?.name ?? DEFAULT_CANDIDATE}]` in workflow.nodes
      ? `train[${candidate?.name ?? DEFAULT_CANDIDATE}]`
      : Object.keys(workflow.graph).find((key) => workflow.graph[key].kind === 'customization_job');
  const node = trainKey ? workflow.nodes[trainKey] : undefined;
  const trained = node?.status === 'succeeded' ? node.outputs.model : undefined;
  if (typeof trained !== 'string') return undefined;
  if (candidate?.finetuning_type === 'all_weights') return trained;
  const student = workflow.inputs.student_model;
  return student ? `${qualify(workflow.workspace, student)}&adapters/${trained}` : undefined;
};

/** Copy an agent config under a new name with every model entry pointed at `model`. */
export const cloneAgentConfig = (
  config: Record<string, unknown>,
  name: string,
  model: string
): Record<string, unknown> => {
  const cloned = structuredClone(config);
  cloned.name = name;
  cloned.config_format = FABRIC_CONFIG_FORMAT;
  if (isRecord(cloned.models)) {
    Object.values(cloned.models).forEach((entry) => {
      if (isRecord(entry)) entry.model = model;
    });
  }
  if (isRecord(cloned.harnesses)) {
    Object.values(cloned.harnesses).forEach((harness) => {
      if (!isRecord(harness)) return;
      if (isRecord(harness.model)) harness.model.model = model;
      else if (typeof harness.model === 'string') harness.model = model;
    });
  }
  return cloned;
};

/** Register a copy of the source agent that runs on the distilled model. */
export const promoteDistillation = async (workflow: DistillationWorkflow) => {
  const model = distilledGatewayModel(workflow);
  if (!model) throw new Error('The workflow has no trained model yet.');

  const [sourceWorkspace, sourceName] = workflow.source_agent.includes('/')
    ? (workflow.source_agent.split('/', 2) as [string, string])
    : ([workflow.workspace, workflow.source_agent] as [string, string]);
  const source = await agentsGetAgent(sourceWorkspace, sourceName);
  if (!source.config || source.config_format !== FABRIC_CONFIG_FORMAT) {
    throw new Error(`Only ${FABRIC_CONFIG_FORMAT} agents with a stored config can be promoted.`);
  }

  const name = `${sourceName}-distilled`;
  return agentsRegisterAgent(workflow.workspace, {
    name,
    description: `${sourceName} on the model distilled by ${workflow.name}.`,
    config: cloneAgentConfig(source.config as Record<string, unknown>, name, model),
    config_format: FABRIC_CONFIG_FORMAT,
    labels: { 'distillation/workflow': workflow.name },
  });
};
