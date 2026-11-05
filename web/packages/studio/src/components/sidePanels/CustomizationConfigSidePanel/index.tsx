// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { Divider, Flex, SidePanel, Stack, Text } from '@nvidia/foundations-react-core';
import { ErrorMessageWithRetry } from '@studio/components/ErrorMessageWithRetry';
import { Loading } from '@studio/components/Layouts/Loading';
import { useCustomizationJob } from '@studio/hooks/useCustomizationJob';
import {
  isAutomodelJob,
  isRlJob,
  isUnslothJob,
  type CustomizationJob,
} from '@studio/util/customizationBackend';
import {
  getBaseModel,
  getFormattedTrainingType,
  getTrainingOptionBadges,
} from '@studio/util/customizations';
import type { ComponentProps, FC, ReactNode } from 'react';

/** Hyperparameter rows specific to each backend. */
const HyperparameterRows: FC<{ job: CustomizationJob }> = ({ job }) => {
  const rows: ReactNode[] = [];
  const { training } = job.spec;
  // RL flattens these onto spec.training under the same names; the other backends
  // split them into spec.schedule / spec.optimizer.
  const schedule = isRlJob(job) ? job.spec.training : job.spec.schedule;
  const optimizer = isRlJob(job) ? job.spec.training : job.spec.optimizer;

  rows.push(<KVPair key="epochs" label="Epochs" value={schedule?.epochs} />);
  rows.push(<KVPair key="max-steps" label="Max Steps" value={schedule?.max_steps} />);
  rows.push(<KVPair key="seed" label="Seed" value={schedule?.seed} />);
  rows.push(<KVPair key="lr" label="Learning Rate" value={optimizer?.learning_rate} />);
  rows.push(<KVPair key="wd" label="Weight Decay" value={optimizer?.weight_decay} />);

  if (isAutomodelJob(job)) {
    const { batch } = job.spec;
    rows.push(<KVPair key="gbs" label="Global Batch Size" value={batch.global_batch_size} />);
    rows.push(<KVPair key="mbs" label="Micro Batch Size" value={batch.micro_batch_size} />);
    rows.push(<KVPair key="warmup" label="Warmup Steps" value={job.spec.optimizer.warmup_steps} />);
    rows.push(<KVPair key="beta1" label="Adam Beta 1" value={job.spec.optimizer.adam_beta1} />);
    rows.push(<KVPair key="beta2" label="Adam Beta 2" value={job.spec.optimizer.adam_beta2} />);
    rows.push(<KVPair key="precision" label="Precision" value={job.spec.training.precision} />);
  }

  if (isUnslothJob(job)) {
    const { batch } = job.spec;
    rows.push(
      <KVPair key="pdbs" label="Per-device Batch Size" value={batch?.per_device_train_batch_size} />
    );
    rows.push(
      <KVPair key="gas" label="Gradient Accumulation" value={batch?.gradient_accumulation_steps} />
    );
    rows.push(<KVPair key="warmup" label="Warmup Steps" value={job.spec.schedule?.warmup_steps} />);
    rows.push(
      <KVPair key="scheduler" label="LR Scheduler" value={job.spec.schedule?.lr_scheduler_type} />
    );
    rows.push(<KVPair key="optim" label="Optimizer" value={job.spec.optimizer?.optim} />);
    rows.push(<KVPair key="precision" label="Precision" value={job.spec.hardware?.precision} />);
  }

  const lora = training && 'lora' in training ? training.lora : undefined;
  if (lora) {
    rows.push(<KVPair key="lora-rank" label="LoRA / Rank" value={lora.rank} />);
    rows.push(<KVPair key="lora-alpha" label="LoRA / Alpha" value={lora.alpha} />);
    rows.push(
      <KVPair
        key="lora-targets"
        label="LoRA / Target Modules"
        value={lora.target_modules?.join(', ')}
      />
    );
  }

  return <>{rows}</>;
};

type Props = ComponentProps<typeof SidePanel> & {
  customizationJobName: string;
  workspace?: string;
};
export const CustomizationConfigSidePanel: FC<Props> = ({
  customizationJobName,
  workspace = '',
  ...attributes
}) => {
  const {
    job,
    isLoading: isLoadingConfig,
    refetch,
  } = useCustomizationJob(workspace, customizationJobName);

  let content;
  if (isLoadingConfig) {
    content = <Loading />;
  } else if (job) {
    const { training } = job.spec;
    // RL discriminates on `type` rather than `training_type`, so these read as blank
    // for DPO and GRPO until the job details work lands.
    const trainingType =
      training && 'training_type' in training ? training.training_type : undefined;
    const finetuningType =
      training && 'finetuning_type' in training ? training.finetuning_type : undefined;
    content = (
      <Stack className="w-full overflow-y-auto" gap="density-xl">
        <KVPair label="Name" value={job.spec.output?.name} />
        <Stack gap="density-sm">
          <Text kind="body/semibold/md">Configuration Snapshot</Text>
          <KVPair label="Base Model" value={getBaseModel(job)} />
          <KVPair label="Training Type" value={getFormattedTrainingType(trainingType)} />
          <KVPair label="Finetuning Type" value={getFormattedTrainingType(finetuningType)} />
          <KVPair
            label="Training Options"
            value={
              <Flex gap="density-sm" wrap="wrap" className="w-full">
                {getTrainingOptionBadges(job)}
              </Flex>
            }
          />
        </Stack>
        <Divider />
        <Stack gap="density-sm">
          <Text kind="body/semibold/md">Hyperparameters</Text>
          <HyperparameterRows job={job} />
        </Stack>
      </Stack>
    );
  } else {
    content = (
      <ErrorMessageWithRetry
        onRetry={refetch}
        message="Failed to fetch customization configuration"
      />
    );
  }
  return (
    <SidePanel
      modal
      bordered
      className="w-[440px]"
      {...attributes}
      slotHeading={<Text kind="label/bold/lg">Customization Configuration</Text>}
    >
      {content}
    </SidePanel>
  );
};
