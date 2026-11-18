// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  Badge,
  Block,
  Checkbox,
  Flex,
  FormField,
  Select,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  DEFAULT_STUDENT_MODEL,
  MOCK_STUDENT_MODELS,
  MOCK_TRACE_EVAL_RUNS,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationData';
import { startMockDistillation } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationWorkflow';
import { getAgentDistillationDetailRoute } from '@studio/routes/utils';
import { type FC, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

interface DistillModalProps {
  open: boolean;
  workspace: string;
  onClose: () => void;
  agentName: string;
}

const MODEL_ITEMS = MOCK_STUDENT_MODELS.map((model) => ({
  value: model.value,
  children: `${model.label} · ${model.params}`,
}));

/**
 * Starts a distillation of the agent's graded trajectories into a smaller model. The runs and
 * models are mocked and Start only confirms the choice; the real job and progress page come next.
 */
export const DistillModal: FC<DistillModalProps> = ({ open, onClose, workspace, agentName }) => {
  const toast = useToast();
  const navigate = useNavigate();
  const [selectedRuns, setSelectedRuns] = useState<Set<string>>(
    () => new Set(MOCK_TRACE_EVAL_RUNS.filter((run) => run.hasTrajectories).map((run) => run.name))
  );
  const [studentModel, setStudentModel] = useState<string | undefined>(DEFAULT_STUDENT_MODEL);

  const exampleCount = useMemo(
    () =>
      MOCK_TRACE_EVAL_RUNS.filter((run) => selectedRuns.has(run.name)).reduce(
        (total, run) => total + run.passedCount,
        0
      ),
    [selectedRuns]
  );

  const toggleRun = (name: string, checked: boolean) =>
    setSelectedRuns((previous) => {
      const next = new Set(previous);
      if (checked) next.add(name);
      else next.delete(name);
      return next;
    });

  const submit = () => {
    if (!studentModel) return;
    const name = `distill-${agentName}-${Date.now().toString(36)}`;
    startMockDistillation({
      name,
      agentName,
      studentModel,
      runCount: selectedRuns.size,
      exampleCount,
    });
    toast.success(`Distillation "${name}" started`);
    onClose();
    navigate(getAgentDistillationDetailRoute(workspace, name));
  };

  return (
    <FormModal
      open={open}
      onClose={onClose}
      className="w-[720px] max-w-[90vw]"
      title="Distill agent"
      instruction="Train a smaller model on this agent's graded runs. Only trials that passed are used as training examples."
      submitButtonText="Start distillation"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      submitDisabled={selectedRuns.size === 0 || !studentModel}
    >
      <FormField slotLabel="Evaluation runs" required>
        <Stack gap="density-sm">
          {MOCK_TRACE_EVAL_RUNS.map((run) => (
            <Flex
              key={run.name}
              align="center"
              justify="between"
              gap="density-md"
              className="rounded-md border border-base p-3"
            >
              <Checkbox
                checked={selectedRuns.has(run.name)}
                disabled={!run.hasTrajectories}
                onChange={(event) => toggleRun(run.name, event.target.checked)}
                slotLabel={run.name}
              />
              <Flex align="center" gap="density-md">
                <Text kind="body/regular/sm" color="secondary">
                  {run.passedCount}/{run.taskCount} passed
                </Text>
                <Badge kind="outline" color={run.hasTrajectories ? 'green' : 'gray'}>
                  {run.hasTrajectories ? 'Trajectories' : 'No trajectories'}
                </Badge>
              </Flex>
            </Flex>
          ))}
        </Stack>
      </FormField>
      <FormField slotLabel="Student model" required>
        <Select
          value={studentModel}
          onValueChange={setStudentModel}
          items={MODEL_ITEMS}
          placeholder="Select a model to distill into"
        />
      </FormField>
      <Block>
        <Text kind="body/regular/sm" color="secondary">
          {selectedRuns.size === 0
            ? 'Select at least one run to see how many training examples it yields.'
            : `${exampleCount} training examples from ${selectedRuns.size} ${selectedRuns.size === 1 ? 'run' : 'runs'}.`}
        </Text>
      </Block>
    </FormModal>
  );
};
