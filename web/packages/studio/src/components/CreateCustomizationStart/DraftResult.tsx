// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { Banner, Button, Flex, Grid, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import type { DraftSetting } from '@studio/components/CreateCustomizationStart/aiDraft';
import type { DraftResultProps } from '@studio/components/CreateCustomizationStart/types';
import { GeneratedConfigPanel } from '@studio/components/CreateFilesetStart/GeneratedConfigPanel';
import { getFormattedTrainingType } from '@studio/util/customizations';
import { FileJson, Pencil } from 'lucide-react';
import { type FC, useState } from 'react';

const PICK_IN_FORM = 'Pick in the form';

const CONFIG_DESCRIPTION =
  'The request the form will submit for this draft, defaults included. Every value can still be changed in the form.';

/** Refs like `workspace/model-name` have no spaces to break on, so they wrap anywhere. */
const WRAP_ANYWHERE = { value: { className: 'break-all' } };

/**
 * The settings people check first, by the last segment of their path — the same name across
 * all three backends' jobs, wherever each one nests it. The rest are in the full config.
 */
const HEADLINE_SETTINGS: Record<string, string> = {
  learning_rate: 'Learning rate',
  epochs: 'Epochs',
  max_steps: 'Max steps',
  global_batch_size: 'Global batch size',
  batch_size: 'Global batch size',
  gradient_accumulation_steps: 'Gradient accumulation',
  rank: 'LoRA rank',
  alpha: 'LoRA alpha',
  warmup_steps: 'Warmup steps',
  micro_batch_size: 'Micro batch size',
  per_device_train_batch_size: 'Per-device batch size',
  max_seq_length: 'Max sequence length',
  num_generations_per_prompt: 'Rollouts per prompt',
  num_gpus_per_node: 'GPUs per node',
  num_nodes: 'Nodes',
  weight_decay: 'Weight decay',
  max_grad_norm: 'Gradient clipping',
  dropout: 'LoRA dropout',
  min_learning_rate: 'Min learning rate',
  warmup_ratio: 'Warmup ratio',
  precision: 'Precision',
  ref_policy_kl_penalty: 'KL penalty',
  distillation_temperature: 'Distillation temperature',
  tensor_parallel_size: 'Tensor parallel size',
};

/** Already stated in the header line, so neither shown nor counted below it. */
const IN_HEADER = new Set(['training.training_type', 'training.finetuning_type', 'training.type']);

const MAX_HEADLINE_SETTINGS = 12;

const headlineLabel = (path: string): string | undefined =>
  HEADLINE_SETTINGS[path.slice(path.lastIndexOf('.') + 1)];

const headlineSettings = (settings: DraftSetting[]): DraftSetting[] =>
  settings
    .flatMap((setting) => {
      const label = headlineLabel(setting.label);
      return label ? [{ ...setting, label }] : [];
    })
    .slice(0, MAX_HEADLINE_SETTINGS);

const BulletList: FC<{ items: string[] }> = ({ items }) => (
  <ul className="list-disc space-y-density-xs pl-density-lg">
    {items.map((item) => (
      <li key={item}>
        <Text kind="body/regular/sm">{item}</Text>
      </li>
    ))}
  </ul>
);

/**
 * A draft that passed the checks, laid out like the job details page's run configuration:
 * what will be trained on the left, why on the right. The left shows only the headline
 * settings; the full request is one click away. Errors never reach here — the panel keeps
 * them on the form, where they can be fixed by regenerating.
 */
export const DraftResult: FC<DraftResultProps> = ({ summary, config, onEdit }) => {
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const settings = summary.settings.filter((setting) => !IN_HEADER.has(setting.label));
  const headline = headlineSettings(settings);
  const hiddenCount = settings.length - headline.length;

  return (
    <Stack gap="density-xl" className="w-full">
      <Flex justify="between" align="start" gap="density-md">
        <Stack gap="density-xs">
          <Text kind="label/bold/lg">{summary.outputName}</Text>
          <Text kind="body/regular/sm" className="text-secondary">
            {[summary.method, summary.finetuningType]
              .filter(Boolean)
              .map(getFormattedTrainingType)
              .concat(summary.backend)
              .join(' · ')}
          </Text>
        </Stack>
        <Flex gap="density-sm" className="shrink-0">
          <Button type="button" kind="tertiary" size="small" onClick={() => setIsConfigOpen(true)}>
            <FileJson size={14} aria-hidden />
            View config
          </Button>
          <Button type="button" kind="secondary" size="small" onClick={onEdit}>
            <Pencil size={14} aria-hidden />
            Edit
          </Button>
        </Flex>
      </Flex>

      <Grid cols={{ base: 1, md: 2 }} gap="density-xl">
        <Panel
          elevation="high"
          className="min-w-0"
          slotHeading={<Text kind="label/bold/lg">Training setup</Text>}
        >
          <Stack gap="density-lg">
            <KVPair
              orientation="vertical"
              label="Base model"
              value={summary.baseModel}
              attributes={WRAP_ANYWHERE}
            />
            <KVPair
              orientation="vertical"
              label="Dataset"
              value={summary.dataset}
              attributes={WRAP_ANYWHERE}
            />
            {summary.method === 'distillation' ? (
              <KVPair orientation="vertical" label="Teacher model" value={PICK_IN_FORM} />
            ) : null}
            {summary.method === 'grpo' ? (
              <KVPair
                orientation="vertical"
                label="Reward environment"
                value={summary.environment ?? PICK_IN_FORM}
                attributes={WRAP_ANYWHERE}
              />
            ) : null}
            {headline.length > 0 ? (
              <Grid cols={2} gap="density-lg">
                {headline.map((setting) => (
                  // Grid cells default to min-width: auto, which lets a long value push
                  // into the next column instead of wrapping.
                  <div key={setting.label} className="min-w-0">
                    <KVPair
                      orientation="vertical"
                      label={setting.label}
                      value={setting.value}
                      attributes={WRAP_ANYWHERE}
                    />
                  </div>
                ))}
              </Grid>
            ) : null}
          </Stack>
          <Text kind="body/regular/xs" className="mt-density-lg text-secondary">
            {hiddenCount > 0
              ? `The draft set ${hiddenCount} more ${hiddenCount === 1 ? 'value' : 'values'}. View config shows the full job.`
              : 'Everything else keeps its default. View config shows the full job.'}
          </Text>
        </Panel>

        <Panel
          elevation="high"
          className="min-w-0"
          slotHeading={<Text kind="label/bold/lg">Why these settings</Text>}
        >
          <Stack gap="density-lg">
            {summary.rationale.length > 0 ? <BulletList items={summary.rationale} /> : null}
            {summary.needsFromUser.length > 0 ? (
              <Banner kind="inline" status="info">
                Before you submit:
                <BulletList items={summary.needsFromUser} />
              </Banner>
            ) : null}
          </Stack>
        </Panel>
      </Grid>

      <GeneratedConfigPanel
        open={isConfigOpen}
        config={config}
        description={CONFIG_DESCRIPTION}
        onClose={() => setIsConfigOpen(false)}
      />
    </Stack>
  );
};
