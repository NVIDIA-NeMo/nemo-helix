// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  FormField,
  SegmentedControl,
  Select,
  Stack,
  Text,
  TextArea,
  TextInput,
} from '@nvidia/foundations-react-core';
import type {
  AgentDeployInput,
  DistillInputs,
  EvalInput,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import {
  useCreateDistillation,
  useRenderDistillation,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationHooks';
import { splitArguments } from '@studio/routes/NewDeploymentRoute/schema';
import { getAgentDistillationDetailRoute } from '@studio/routes/utils';
import { type FC, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

interface DistillModalProps {
  open: boolean;
  workspace: string;
  onClose: () => void;
  agentName: string;
}

type TrajectorySource = 'teacher' | 'trajectories';

const SOURCE_ITEMS = [
  { value: 'teacher', children: 'Run the teacher' },
  { value: 'trajectories', children: 'Use existing trajectories' },
];

const EVAL_KIND_ITEMS = [
  { value: 'agent', children: 'Agent evaluation (tasks)' },
  { value: 'dataset', children: 'Dataset evaluation' },
];

const AGENT_MODES = ['k8s', 'docker', 'openshell', 'subprocess'] as const;

const AGENT_MODE_ITEMS = [
  { value: 'k8s', children: 'Kubernetes' },
  { value: 'docker', children: 'Docker' },
  { value: 'openshell', children: 'OpenShell' },
  { value: 'subprocess', children: 'Subprocess' },
];

const FINETUNING_ITEMS = [
  { value: 'lora', children: 'LoRA' },
  { value: 'all_weights', children: 'Full fine-tune' },
];

const positiveInt = (text: string): number | undefined => {
  const value = Number(text);
  return Number.isInteger(value) && value > 0 ? value : undefined;
};

const splitList = (text: string): string[] =>
  text
    .split(/[\s,]+/)
    .map((entry) => entry.trim())
    .filter(Boolean);

/**
 * Starts a distillation workflow. The preview comes from the render endpoint, which writes nothing,
 * so it updates as the form changes.
 */
export const DistillModal: FC<DistillModalProps> = ({ open, onClose, workspace, agentName }) => {
  const toast = useToast();
  const navigate = useNavigate();
  const create = useCreateDistillation(workspace);

  const [source, setSource] = useState<TrajectorySource>('teacher');
  const [teacherModel, setTeacherModel] = useState('');
  const [trajectories, setTrajectories] = useState('');
  const [studentModel, setStudentModel] = useState('');
  const [evalName, setEvalName] = useState('holdout');
  const [evalKind, setEvalKind] = useState<EvalInput['kind']>('agent');
  const [evalConfig, setEvalConfig] = useState('');
  const [agentMode, setAgentMode] = useState<NonNullable<AgentDeployInput['mode']>>('k8s');
  const [agentImage, setAgentImage] = useState('');
  const [finetuningType, setFinetuningType] = useState<'lora' | 'all_weights'>('lora');
  const [trainingGpus, setTrainingGpus] = useState('1');
  const [servingGpus, setServingGpus] = useState('1');
  const [gpuBudget, setGpuBudget] = useState('8');
  const [servingArgs, setServingArgs] = useState('');

  const inputs = useMemo<DistillInputs | undefined>(() => {
    const servingArguments = splitArguments(servingArgs);
    const training = positiveInt(trainingGpus);
    const serving = positiveInt(servingGpus);
    const budget = positiveInt(gpuBudget);
    const trajectoryList = splitList(trajectories);
    const needsImage = agentMode !== 'subprocess';
    if (
      !studentModel.trim() ||
      !evalName.trim() ||
      !evalConfig.trim() ||
      !servingArguments ||
      !training ||
      !serving ||
      !budget ||
      (needsImage && !agentImage.trim()) ||
      (source === 'teacher' ? !teacherModel.trim() : trajectoryList.length === 0)
    ) {
      return undefined;
    }
    return {
      source_agent: `${workspace}/${agentName}`,
      student_model: studentModel.trim(),
      ...(source === 'teacher'
        ? { teacher: { model: teacherModel.trim(), gpu: serving, additional_args: servingArguments } }
        : { trajectory_filesets: trajectoryList }),
      gpu_budget: budget,
      agent_deployment: { mode: agentMode, image: agentImage.trim() },
      evals: [{ name: evalName.trim(), kind: evalKind, config: evalConfig.trim() }],
      candidates: [{ name: finetuningType === 'lora' ? 'lora' : 'full', finetuning_type: finetuningType, gpu: training }],
      deployment: { engine: 'vllm', gpu: serving, additional_args: servingArguments },
    };
  }, [
    workspace,
    agentName,
    source,
    teacherModel,
    trajectories,
    studentModel,
    evalName,
    evalKind,
    evalConfig,
    agentMode,
    agentImage,
    finetuningType,
    trainingGpus,
    servingGpus,
    gpuBudget,
    servingArgs,
  ]);

  const { data: preview, error: previewError } = useRenderDistillation(workspace, inputs);

  const submit = () => {
    if (!inputs) return;
    create.mutate(
      { inputs },
      {
        onSuccess: (workflow) => {
          toast.success(`Distillation "${workflow.name}" started`);
          onClose();
          navigate(getAgentDistillationDetailRoute(workspace, workflow.name));
        },
      }
    );
  };

  const summary = previewError
    ? getErrorMessage(previewError)
    : preview
      ? `${preview.counts.nodes} steps · ${preview.counts.candidates} candidate${preview.counts.candidates === 1 ? '' : 's'} · ${preview.counts.passing_trials} passing trials · ~${preview.counts.gpu_hours} GPU hours · ${preview.counts.parallel_jobs} parallel job${preview.counts.parallel_jobs === 1 ? '' : 's'}`
      : 'Fill in the required fields to preview the workflow.';

  return (
    <FormModal
      open={open}
      onClose={onClose}
      className="w-[760px] max-w-[90vw]"
      title="Distill agent"
      instruction="Train a smaller model on this agent's graded runs, then compare it with the original and with a plain model swap on a held-out evaluation."
      submitButtonText="Start distillation"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      disabled={create.isPending}
      loading={create.isPending}
      submitDisabled={!inputs}
      errorText={create.error ? getErrorMessage(create.error) : undefined}
    >
      <Stack gap="density-lg">
        <FormField slotLabel="Training data" required>
          <SegmentedControl
            className="w-fit"
            aria-label="Training data source"
            value={source}
            onValueChange={(next) => setSource(next === 'trajectories' ? 'trajectories' : 'teacher')}
            items={SOURCE_ITEMS}
          />
        </FormField>
        {source === 'teacher' ? (
          <FormField
            slotLabel="Teacher model"
            slotInfo="The workflow deploys this model and runs the original evaluations on it to produce trajectories."
            required
          >
            <TextInput
              value={teacherModel}
              placeholder="workspace/model"
              onChange={(event) => setTeacherModel(event.currentTarget.value)}
            />
          </FormField>
        ) : (
          <FormField
            slotLabel="Trajectory filesets"
            slotInfo="Evaluation output filesets to learn from, as workspace/name, separated by commas or spaces."
            required
          >
            <TextInput
              value={trajectories}
              placeholder="workspace/eval-output"
              onChange={(event) => setTrajectories(event.currentTarget.value)}
            />
          </FormField>
        )}
        <FormField slotLabel="Student model" slotInfo="The smaller model to train." required>
          <TextInput
            value={studentModel}
            placeholder="workspace/model"
            onChange={(event) => setStudentModel(event.currentTarget.value)}
          />
        </FormField>
        <FormField slotLabel="Evaluation" required>
          <Stack gap="density-sm">
            <TextInput
              value={evalName}
              placeholder="Name"
              onChange={(event) => setEvalName(event.currentTarget.value)}
            />
            <Select
              value={evalKind}
              onValueChange={(next) => setEvalKind(next === 'dataset' ? 'dataset' : 'agent')}
              items={EVAL_KIND_ITEMS}
            />
            <TextInput
              value={evalConfig}
              placeholder="workspace/fileset#path/to/eval-config.json"
              onChange={(event) => setEvalConfig(event.currentTarget.value)}
            />
          </Stack>
        </FormField>
        <FormField
          slotLabel="Agent deployment"
          slotInfo="How the workflow deploys the agent for each evaluation. Container modes need an image."
          required
        >
          <Stack gap="density-sm">
            <Select
              value={agentMode}
              onValueChange={(next) => setAgentMode(AGENT_MODES.find((mode) => mode === next) ?? 'k8s')}
              items={AGENT_MODE_ITEMS}
            />
            <TextInput
              value={agentImage}
              placeholder="registry/agent-image:tag"
              onChange={(event) => setAgentImage(event.currentTarget.value)}
            />
          </Stack>
        </FormField>
        <FormField slotLabel="Training" required>
          <Stack gap="density-sm">
            <Select
              value={finetuningType}
              onValueChange={(next) => setFinetuningType(next === 'all_weights' ? 'all_weights' : 'lora')}
              items={FINETUNING_ITEMS}
            />
            <TextInput
              value={trainingGpus}
              placeholder="GPUs per training job"
              onChange={(event) => setTrainingGpus(event.currentTarget.value)}
            />
          </Stack>
        </FormField>
        <FormField
          slotLabel="Serving"
          slotInfo="GPUs for each model deployment, and extra vLLM arguments appended to its command."
          required
        >
          <Stack gap="density-sm">
            <TextInput
              value={servingGpus}
              placeholder="GPUs per deployment"
              onChange={(event) => setServingGpus(event.currentTarget.value)}
            />
            <TextArea
              value={servingArgs}
              rows={4}
              placeholder={'--tokenizer-mode glm5_next\n--tool-call-parser glm47\n--enable-auto-tool-choice'}
              onValueChange={setServingArgs}
            />
          </Stack>
        </FormField>
        <FormField
          slotLabel="GPU budget"
          slotInfo="GPUs the workflow may use at once. Candidate jobs run in parallel lanes within it."
          required
        >
          <TextInput value={gpuBudget} onChange={(event) => setGpuBudget(event.currentTarget.value)} />
        </FormField>
        <Text kind="body/regular/sm" color="secondary">
          {summary}
        </Text>
      </Stack>
    </FormModal>
  );
};
