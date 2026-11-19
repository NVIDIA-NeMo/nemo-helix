// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsGetAgent, agentsRegisterAgent } from '@nemo/sdk/generated/agents/agents';
import type { DistillationWorkflow } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import { DISTILLATION_MOCK } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationMock';

const FABRIC_CONFIG_FORMAT = 'nemo-agents-spec-v1';

const qualify = (workspace: string, model: string): string =>
  model.includes('/') ? model : `${workspace}/${model}`;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * The gateway name of the trained model, built the way the plugin names the arm it evaluates: a LoRA
 * candidate is an adapter on the student, a full fine-tune is a model of its own.
 */
export const distilledGatewayModel = (
  workflow: DistillationWorkflow,
  candidateName: string
): string | undefined => {
  const candidate = workflow.inputs.candidates?.find((entry) => entry.name === candidateName);
  const trainKey = Object.keys(workflow.graph).find(
    (key) => workflow.graph[key].kind === 'customization_job' && key.endsWith(`[${candidateName}]`)
  );
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
export const promoteDistillation = async (
  workflow: DistillationWorkflow,
  candidateName: string
): Promise<{ name: string }> => {
  if (DISTILLATION_MOCK) {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    return { name: `${workflow.source_agent.split('/').pop()}-distilled-${candidateName}` };
  }
  const model = distilledGatewayModel(workflow, candidateName);
  if (!model) throw new Error(`Candidate ${candidateName} has no trained model yet.`);

  const [sourceWorkspace, sourceName] = workflow.source_agent.includes('/')
    ? (workflow.source_agent.split('/', 2) as [string, string])
    : ([workflow.workspace, workflow.source_agent] as [string, string]);
  const source = await agentsGetAgent(sourceWorkspace, sourceName);
  if (!source.config || source.config_format !== FABRIC_CONFIG_FORMAT) {
    throw new Error(`Only ${FABRIC_CONFIG_FORMAT} agents with a stored config can be promoted.`);
  }

  const multiple = (workflow.inputs.candidates?.length ?? 1) > 1;
  const name = multiple ? `${sourceName}-distilled-${candidateName}` : `${sourceName}-distilled`;
  const agent = await agentsRegisterAgent(workflow.workspace, {
    name,
    description: `${sourceName} on the model distilled by ${workflow.name} (${candidateName}).`,
    config: cloneAgentConfig(source.config as Record<string, unknown>, name, model),
    config_format: FABRIC_CONFIG_FORMAT,
    labels: { 'distillation/workflow': workflow.name, 'distillation/candidate': candidateName },
  });
  return { name: agent.name ?? name };
};
