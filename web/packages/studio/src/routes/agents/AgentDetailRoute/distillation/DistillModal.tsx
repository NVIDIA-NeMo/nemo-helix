// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { useModelsListModels } from '@nemo/sdk/generated/platform/models';
import {
  FormField,
  Select,
  Spinner,
  Text,
} from '@nvidia/foundations-react-core';
import type {
  AgentDeployInput,
  DistillInputs,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import {
  DISTILLATION_DEFAULTS,
  PREFERRED_STUDENT_MODEL,
  servingArgsFor,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationDefaults';
import { useCreateDistillation } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationHooks';
import {
  DISTILLATION_MOCK,
  MOCK_MODEL_NAMES,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationMock';
import {
  trajectoryFilesetOf,
  useAgentDeploymentImage,
  useAgentEvalRuns,
  writeEvalConfig,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationRuns';
import { getAgentDistillationDetailRoute } from '@studio/routes/utils';
import { type FC, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

interface DistillModalProps {
  open: boolean;
  workspace: string;
  onClose: () => void;
  agentName: string;
}

const CONTAINER_MODES: ReadonlySet<string> = new Set(['docker', 'k8s', 'openshell']);

const isContainerMode = (mode: string): mode is NonNullable<AgentDeployInput['mode']> =>
  CONTAINER_MODES.has(mode);

/**
 * Starts a distillation of the agent's evaluation runs into a smaller model. The trajectories come from
 * the runs picked here, and the same runs' tasks are what every model is scored on. Everything else
 * is set from defaults.
 */
export const DistillModal: FC<DistillModalProps> = ({ open, onClose, workspace, agentName }) => {
  const toast = useToast();
  const navigate = useNavigate();
  const create = useCreateDistillation(workspace);
  const { data: runs = [], isLoading: isLoadingRuns } = useAgentEvalRuns(workspace, agentName);
  const { data: models } = useModelsListModels(workspace, { page_size: 100 });
  const { data: image } = useAgentDeploymentImage(workspace, agentName);
  const [chosenRun, setChosenRun] = useState<string>();
  const [chosenModel, setChosenModel] = useState<string>();
  const [isStarting, setIsStarting] = useState(false);

  const modelItems = useMemo(
    () =>
      DISTILLATION_MOCK
        ? MOCK_MODEL_NAMES.map((name) => ({ value: `${workspace}/${name}`, children: name }))
        : (models?.data ?? []).map((model) => ({
            value: `${model.workspace}/${model.name}`,
            children: model.name,
          })),
    [models, workspace]
  );
  const studentModel =
    chosenModel ??
    modelItems.find((item) => PREFERRED_STUDENT_MODEL.test(item.value))?.value ??
    modelItems[0]?.value;

  const runItems = useMemo(
    () =>
      runs.map((run) => ({
        value: run.name,
        children: `${run.name} · ${run.taskCount} ${run.taskCount === 1 ? 'task' : 'tasks'}${
          run.createdAt ? ` · ${new Date(run.createdAt).toLocaleDateString()}` : ''
        }`,
      })),
    [runs]
  );
  const selectedRun = runs.find((run) => run.name === (chosenRun ?? runs[0]?.name));

  const submit = async () => {
    if (!studentModel || !selectedRun) return;
    setIsStarting(true);
    try {
      const [trajectoryFileset, evalConfig] = await Promise.all([
        trajectoryFilesetOf(workspace, selectedRun.name),
        writeEvalConfig(workspace, agentName, [selectedRun]),
      ]);
      const inputs: DistillInputs = {
        source_agent: `${workspace}/${agentName}`,
        student_model: studentModel,
        trajectory_filesets: [trajectoryFileset],
        gpu_budget: DISTILLATION_DEFAULTS.gpuBudget,
        agent_deployment:
          image && isContainerMode(image.mode)
            ? { mode: image.mode, image: image.image }
            : { mode: 'subprocess', image: '' },
        evals: [{ name: 'tasks', kind: 'agent', config: evalConfig }],
        candidates: [
          {
            name: DISTILLATION_DEFAULTS.candidateName,
            finetuning_type: 'lora',
            gpu: DISTILLATION_DEFAULTS.trainingGpus,
          },
        ],
        deployment: {
          engine: 'vllm',
          gpu: DISTILLATION_DEFAULTS.servingGpus,
          additional_args: servingArgsFor(studentModel),
        },
      };
      const workflow = await create.mutateAsync({ inputs });
      toast.success(`Distillation "${workflow.name}" started`);
      onClose();
      navigate(getAgentDistillationDetailRoute(workspace, workflow.name));
    } catch (error) {
      toast.error(error instanceof Error ? getErrorMessage(error) : 'Failed to start the distillation');
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <FormModal
      open={open}
      onClose={onClose}
      className="w-[720px] max-w-[90vw]"
      title="Distill agent"
      instruction="Train a smaller model on one of this agent's evaluation runs. Only trials that passed are used as training examples."
      submitButtonText="Start distillation"
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
      submitDisabled={!selectedRun || !studentModel || isStarting}
    >
      <FormField
        slotLabel="Evaluation run"
        slotInfo="Its trajectories train the student, and its tasks are what the original agent, the untrained student and the distilled student are compared on."
        required
      >
        {isLoadingRuns ? (
          <Spinner size="small" aria-label="Loading evaluation runs" />
        ) : runs.length === 0 ? (
          <Text kind="body/regular/sm" color="secondary">
            This agent has no completed evaluation runs yet.
          </Text>
        ) : (
          <Select
            value={selectedRun?.name}
            onValueChange={setChosenRun}
            items={runItems}
            placeholder="Select an evaluation run"
          />
        )}
      </FormField>
      <FormField slotLabel="Student model" required>
        <Select
          value={studentModel}
          onValueChange={setChosenModel}
          items={modelItems}
          placeholder="Select a model to distill into"
        />
      </FormField>
    </FormModal>
  );
};
