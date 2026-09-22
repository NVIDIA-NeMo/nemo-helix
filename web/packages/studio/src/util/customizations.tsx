// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StatTileDiagnosticProps, StatTileStatus } from '@nemo/common/src/components/StatTile';
import { formatTimeInSeconds, utcToLocalDate } from '@nemo/common/src/utils/date';
import { formatFinetuningType } from '@nemo/common/src/utils/formatters';
import type { RlGRPOTraining, RlJobOutput } from '@nemo/sdk/generated/customizer/schema';
import type { PlatformJobStatus } from '@nemo/sdk/generated/platform/schema';
import { Badge } from '@nvidia/foundations-react-core';
import type {
  CustomizationMetricValue,
  CustomizationStatusDetailsWithMetrics,
  CustomizationTrainingTelemetry,
} from '@studio/types/customization';
import {
  isAutomodelJob,
  isGrpoJob,
  isRlJob,
  isUnslothJob,
  type CustomizationJob,
} from '@studio/util/customizationBackend';
import { formatElapsedTime } from '@studio/util/date';
import { formatStepDuration, GRPO_METRIC, medianValue, readSeries } from '@studio/util/grpoMetrics';
import { getTextWithCount } from '@studio/util/strings';
import { Circle /* TODO: replace with a proper icon (was Circle) */, Gpu } from 'lucide-react';
import { ReactNode } from 'react';

export { formatFinetuningType };

export type FileType = 'training' | 'testing' | 'validation';

/** Training/finetuning type for display (e.g. training.training_type / training.finetuning_type). */
export const getFormattedTrainingType = (type?: string) => {
  if (type === undefined) {
    return '';
  }
  switch (type) {
    case 'lora': {
      return 'LoRA';
    }
    case 'lora_merged': {
      return 'LoRA (merged)';
    }
    case 'all_weights': {
      return 'All Weights';
    }
    case 'sft': {
      return 'SFT';
    }
    case 'distillation': {
      return 'Distillation';
    }
    default: {
      return type;
    }
  }
};

/**
 * Returns the given status formatted in title case. For example, DEPLOYMENT_IN_PROGRESS returns
 * 'Deployment In Progress', optionally with the progress percentage.
 */
export const getFormattedCustomizationStatus = (
  status?: PlatformJobStatus | string,
  progressPercent?: number
) => {
  let statusText = '';

  if (status) {
    statusText = status
      .split('_')
      .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  }

  if (progressPercent !== undefined) {
    statusText += ` (${Math.floor(progressPercent)}%)`;
  }

  return statusText;
};

/**
 * Returns the base model reference for a customization job. Automodel stores it as a string;
 * unsloth stores it as a `ModelLoadSpec` whose `name` is the reference.
 */
export const getBaseModel = (customizationJob?: CustomizationJob): string => {
  if (!customizationJob) {
    return '';
  }
  if (isUnslothJob(customizationJob)) {
    return customizationJob.spec.model.name ?? '';
  }
  if (isAutomodelJob(customizationJob)) {
    return customizationJob.spec.model ?? '';
  }
  if (isRlJob(customizationJob)) {
    return customizationJob.spec.model ?? '';
  }
  return '';
};

export const getFinetuningType = (customizationJob?: CustomizationJob): string => {
  if (!customizationJob) {
    return '';
  }
  if (isAutomodelJob(customizationJob)) {
    return customizationJob.spec.training?.finetuning_type ?? '';
  }
  if (isUnslothJob(customizationJob)) {
    return customizationJob.spec.training?.finetuning_type ?? '';
  }
  // GRPO carries `finetuning_type` like the other backends; DPO does not, hence the narrowing.
  if (isGrpoJob(customizationJob)) {
    return customizationJob.spec.training.finetuning_type ?? '';
  }
  return '';
};

/**
 * Returns the training dataset URI for a customization job. Automodel stores it under
 * `dataset.training`; unsloth stores it under `dataset.path`.
 */
export const getDatasetUri = (customizationJob?: CustomizationJob): string => {
  if (!customizationJob) {
    return '';
  }
  if (isAutomodelJob(customizationJob)) {
    return customizationJob.spec.dataset.training ?? '';
  }
  if (isUnslothJob(customizationJob)) {
    return customizationJob.spec.dataset.path ?? '';
  }
  // A bare fileset ref, not a per-split object: GRPO trains on prompts and the env scores them.
  if (isRlJob(customizationJob)) {
    return customizationJob.spec.dataset ?? '';
  }
  return '';
};

/**
 * Effective training batch size, used to compute the loss-chart x-axis. Automodel uses
 * `batch.global_batch_size`; unsloth uses `batch.per_device_train_batch_size`.
 */
export const getTrainingBatchSize = (customizationJob?: CustomizationJob): number => {
  if (!customizationJob) {
    return 0;
  }
  if (isAutomodelJob(customizationJob)) {
    return customizationJob.spec.batch.global_batch_size ?? 0;
  }
  if (isUnslothJob(customizationJob)) {
    return customizationJob.spec.batch?.per_device_train_batch_size ?? 0;
  }
  return 0;
};

/**
 * Returns a string that represents the number of epochs completed by the given customization.
 */
export const getCustomizationTrainingProgress = (customization: CustomizationJob) => {
  if (!customization.status_details) {
    return '';
  }

  // RL keeps epochs on spec.training; the other backends use spec.schedule.
  const epochs = isRlJob(customization)
    ? customization.spec?.training?.epochs
    : customization.spec?.schedule?.epochs;

  const { epoch, percentage_done: percentageDone } = customization.status_details || {};

  if (epoch == null && percentageDone == null) {
    return '';
  }

  return `${epoch ?? 0}/${epochs ?? '?'} (${Math.floor(Number(percentageDone) || 0)}%)`;
};

const asFiniteNumber = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;

const asNonEmptyString = (value: unknown): string | undefined =>
  typeof value === 'string' && value.length > 0 ? value : undefined;

export const getTrainingTelemetry = (
  job: CustomizationJob | null | undefined
): CustomizationTrainingTelemetry => {
  const details = job?.status_details;
  if (!details) return {};
  return {
    phase: asNonEmptyString(details.phase),
    step: asFiniteNumber(details.step),
    maxSteps: asFiniteNumber(details.max_steps),
    numEpochs: asFiniteNumber(details.num_epochs),
    epoch: asFiniteNumber(details.epoch),
    trainLoss: asFiniteNumber(details.train_loss),
    valLoss: asFiniteNumber(details.val_loss),
    // `?? details.lr` reads jobs written before the phase prefix was applied to
    // every metric name. Those are already in the database and never change, so
    // without the fallback their Learning Rate and Gradient Norm render blank
    // forever. `train_loss` and `val_loss` need no equivalent -- they were
    // already spelled that way.
    learningRate: asFiniteNumber(details.train_lr ?? details.lr),
    gradNorm: asFiniteNumber(details.train_grad_norm ?? details.grad_norm),
    checkpointPath: asNonEmptyString(details.checkpoint_path),
  };
};

interface MetricSummary {
  final: number;
  deltaFromStart?: number;
}

const summarizeMetric = (series?: CustomizationMetricValue[]): MetricSummary | undefined => {
  if (!series?.length) return undefined;
  const final = series[series.length - 1].value;
  const deltaFromStart = series.length > 1 ? final - series[0].value : undefined;
  return { final, deltaFromStart };
};

/**
 * Past this magnitude `toFixed` produces a string long enough to distort the tile grid, so the
 * value switches to exponential. Diverged runs are exactly when these numbers get large.
 */
const EXPONENTIAL_ABOVE = 1e6;

/**
 * Longest phase label rendered before truncating. Known phases top out at "Processing
 * Checkpoint" (21 chars); this only bites on phases the backend adds later, which are
 * title-cased verbatim and otherwise unbounded.
 */
const MAX_PHASE_LENGTH = 22;

const truncatePhase = (phase: string): string =>
  phase.length > MAX_PHASE_LENGTH ? `${phase.slice(0, MAX_PHASE_LENGTH - 1)}…` : phase;

export const formatMetricValue = (value: number, decimals = 4): string =>
  Math.abs(value) >= EXPONENTIAL_ABOVE ? value.toExponential(2) : value.toFixed(decimals);

interface FormatStepCountOptions {
  /** Value past which compact notation ("10K") replaces the plain, comma-grouped one. */
  floor?: number;
  /** Lowercases the compact suffix ("10k" rather than "10K"), for prose rather than a tile. */
  lowercase?: boolean;
}

/** Compact past four digits so a long run's step count cannot widen the column. */
export const formatStepCount = (
  value: number,
  { floor = 10_000, lowercase = false }: FormatStepCountOptions = {}
): string => {
  if (Math.abs(value) < floor) {
    return value.toLocaleString();
  }
  const compact = new Intl.NumberFormat(undefined, {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value);
  return lowercase ? compact.toLowerCase() : compact;
};

const formatDeltaHint = (delta: number, decimals: number): string =>
  `${delta >= 0 ? '+' : ''}${formatMetricValue(delta, decimals)} from start`;

const NOT_AVAILABLE = '—';

interface MetricTileOptions {
  /** Which direction of travel is good news, and so which way the delta is tinted. */
  betterWhen: 'lower' | 'higher';
  formatValue?: (value: number) => string;
  hint?: string;
}

/**
 * `betterWhen` is not cosmetic: loss falling is progress, reward falling is the run going
 * backwards, so assuming one direction would tint the other's delta exactly wrong.
 */
const metricTile = (
  label: string,
  summary: MetricSummary | undefined,
  { betterWhen, formatValue = formatMetricValue, hint }: MetricTileOptions
): StatTileDiagnosticProps => {
  if (!summary) {
    return { label, value: NOT_AVAILABLE, hint };
  }
  const { deltaFromStart } = summary;
  const value = formatValue(summary.final);
  if (deltaFromStart === undefined) {
    return { label, value, hint };
  }

  // No movement is neither good nor bad; tinting a stalled reward red reads as a failure.
  const status: StatTileStatus =
    deltaFromStart === 0
      ? 'neutral'
      : (betterWhen === 'lower' ? deltaFromStart < 0 : deltaFromStart > 0)
        ? 'success'
        : 'error';
  const delta = formatDeltaHint(deltaFromStart, 4);

  // The delta only takes the hint slot when the tile has no context to put there instead.
  return hint
    ? { label, value, trailingLabel: delta, trailingLabelStatus: status, hint }
    : { label, value, hint: delta, hintStatus: status };
};

export const getTrainingProgressTiles = (
  telemetry: CustomizationTrainingTelemetry
): StatTileDiagnosticProps[] => [
  {
    label: 'Steps Completed',
    value:
      telemetry.step === undefined
        ? NOT_AVAILABLE
        : telemetry.maxSteps
          ? `${formatStepCount(telemetry.step)} / ${formatStepCount(telemetry.maxSteps)}`
          : formatStepCount(telemetry.step),
  },
  {
    label: 'Epochs Completed',
    value:
      telemetry.epoch === undefined
        ? NOT_AVAILABLE
        : telemetry.numEpochs
          ? `${telemetry.epoch} / ${telemetry.numEpochs}`
          : String(telemetry.epoch),
  },
];

export const getLossTiles = (
  statusDetails: CustomizationStatusDetailsWithMetrics | undefined,
  isTerminal = false
): StatTileDiagnosticProps[] => [
  metricTile(
    `${isTerminal ? 'Final' : 'Latest'} Training Loss`,
    summarizeMetric(statusDetails?.metrics?.train_loss),
    { betterWhen: 'lower' }
  ),
  metricTile(
    `${isTerminal ? 'Final' : 'Latest'} Validation Loss`,
    summarizeMetric(statusDetails?.metrics?.val_loss),
    { betterWhen: 'lower' }
  ),
];

const TRAINING_PHASE_LABELS: Record<string, string> = {
  compiling_config: 'Compiling Config',
  automodel_recipe_setup: 'Recipe Setup',
  training: 'Training',
  validation: 'Validation',
  checkpoint_saved: 'Checkpoint Saved',
  epoch_end: 'Epoch End',
  processing_checkpoint: 'Processing Checkpoint',
  completed: 'Completed',
};

export const formatTrainingPhase = (phase: string): string =>
  TRAINING_PHASE_LABELS[phase] ??
  phase
    .split('_')
    .map((word) => (word ? word.charAt(0).toUpperCase() + word.slice(1) : word))
    .join(' ');

export interface JobDurationStep {
  created_at: string;
  updated_at: string;
}

export const getJobStartDate = (steps: readonly JobDurationStep[] | undefined): Date | undefined =>
  steps?.length ? utcToLocalDate(steps[0].created_at) : undefined;

export const getJobDuration = (
  steps: readonly JobDurationStep[] | undefined,
  isTerminal: boolean,
  liveSeconds?: number
): string => {
  if (!isTerminal) {
    return formatTimeInSeconds(liveSeconds) || NOT_AVAILABLE;
  }

  if (!steps?.length) {
    return NOT_AVAILABLE;
  }

  const start = utcToLocalDate(steps[0].created_at);
  const end = utcToLocalDate(steps[steps.length - 1].updated_at);
  if (!start || !end || end.getTime() - start.getTime() < 1000) {
    return NOT_AVAILABLE;
  }
  return formatElapsedTime(start, end);
};

interface TrainingDiagnosticsContext {
  isTerminal: boolean;
  duration: string;
  /** Display name of the failing pipeline step, when the job errored. */
  failedAtStepLabel?: string;
}

/** Where the run ended up: failing step, total time once finished, or current stage while running. */
const getRunStateTile = (
  telemetry: CustomizationTrainingTelemetry,
  { isTerminal, duration, failedAtStepLabel }: TrainingDiagnosticsContext
): StatTileDiagnosticProps => {
  if (failedAtStepLabel) {
    return {
      label: 'Run State',
      value: 'Failed',
      hint: `during ${failedAtStepLabel}`,
      // Every panel renders the run-state tile unbordered, and StatTile's border tint only
      // applies to the bordered branch — so `hintStatus` is what actually reads as failure here.
      hintStatus: 'error',
      status: 'error',
    };
  }
  return isTerminal
    ? { label: 'Duration', value: duration, hint: 'total run time' }
    : {
        label: 'Phase',
        value: telemetry.phase
          ? truncatePhase(formatTrainingPhase(telemetry.phase))
          : NOT_AVAILABLE,
        hint: 'current stage',
      };
};

export const getTrainingDiagnosticsTiles = (
  telemetry: CustomizationTrainingTelemetry,
  statusDetails: CustomizationStatusDetailsWithMetrics | undefined,
  context: TrainingDiagnosticsContext
): StatTileDiagnosticProps[] => {
  const trainLoss = summarizeMetric(statusDetails?.metrics?.train_loss);
  const valLoss = summarizeMetric(statusDetails?.metrics?.val_loss);

  return [
    {
      label: 'Learning Rate',
      value: telemetry.learningRate?.toExponential(2) ?? NOT_AVAILABLE,
      hint: 'at latest step',
    },
    {
      label: 'Gradient Norm',
      value:
        telemetry.gradNorm !== undefined ? formatMetricValue(telemetry.gradNorm) : NOT_AVAILABLE,
      hint: 'at latest step',
    },
    {
      label: 'Train/Val Gap',
      value:
        trainLoss && valLoss ? formatMetricValue(valLoss.final - trainLoss.final) : NOT_AVAILABLE,
      hint: 'validation - training',
    },
    getRunStateTile(telemetry, context),
  ];
};

const formatPercent = (value: number): string => `${(value * 100).toFixed(1)}%`;

/**
 * Loss is deliberately absent: GRPO's is a policy-gradient surrogate whose magnitude means
 * nothing, and it reports no validation loss at all.
 */
export const getGrpoSummaryTiles = (
  statusDetails: CustomizationStatusDetailsWithMetrics | undefined,
  isTerminal = false
): StatTileDiagnosticProps[] => {
  const validation = readSeries(statusDetails, GRPO_METRIC.validationReward);
  const lastEvalStep = validation?.[validation.length - 1]?.step;
  const truncation = summarizeMetric(readSeries(statusDetails, GRPO_METRIC.truncationRate));
  const stepTime = medianValue(readSeries(statusDetails, GRPO_METRIC.stepTime));

  return [
    metricTile(
      `${isTerminal ? 'Final' : 'Latest'} Mean Reward`,
      summarizeMetric(readSeries(statusDetails, GRPO_METRIC.trainReward)),
      { betterWhen: 'higher', hint: 'across all sampled rollouts' }
    ),
    metricTile('Validation Reward', summarizeMetric(validation), {
      betterWhen: 'higher',
      hint:
        lastEvalStep === undefined
          ? 'held-out prompts'
          : `held-out prompts, step ${formatStepCount(lastEvalStep)}`,
    }),
    {
      label: 'Median Step Time',
      value: stepTime === undefined ? NOT_AVAILABLE : formatStepDuration(stepTime),
      hint: 'wall clock per step',
    },
    {
      label: 'Truncation Rate',
      value: truncation ? formatPercent(truncation.final) : NOT_AVAILABLE,
      hint: 'hit the length limit',
    },
  ];
};

/**
 * Rollouts per training step: group size (rollouts per prompt) times prompts sampled per step.
 * When the spec omits either factor, the service derives it from `batch_size` divided by the
 * other factor, so the product collapses back to `batch_size` directly rather than dividing and
 * multiplying back through a possibly-imprecise fraction.
 */
const getGrpoRolloutsPerStep = (training: RlGRPOTraining): number =>
  training.num_prompts_per_step !== undefined && training.num_generations_per_prompt !== undefined
    ? training.num_generations_per_prompt * training.num_prompts_per_step
    : (training.batch_size ?? 0);

/**
 * Sub-header summary for a GRPO run: which training step it's on (out of the planned total, when
 * known — whether still running or stopped short of it), or how many steps ran once terminal with
 * no known target, plus the rollouts that implies. Rollout counts aren't reported directly, so
 * they're derived from the step count and the group-size/prompts-per-step spec.
 */
export const getGrpoRunProgressSummary = (
  customization: CustomizationJob | undefined,
  telemetry: CustomizationTrainingTelemetry,
  isTerminal: boolean
): string => {
  if (!customization || !isGrpoJob(customization) || telemetry.step === undefined) {
    return '';
  }

  const stepText =
    telemetry.maxSteps !== undefined
      ? `step ${formatStepCount(telemetry.step)} of ${formatStepCount(telemetry.maxSteps)}`
      : isTerminal
        ? `${formatStepCount(telemetry.step)} ${telemetry.step === 1 ? 'step' : 'steps'} ran`
        : `step ${formatStepCount(telemetry.step)}`;

  const rolloutsPerStep = getGrpoRolloutsPerStep(customization.spec.training);
  const rolloutsText =
    rolloutsPerStep > 0
      ? `${formatStepCount(telemetry.step * rolloutsPerStep, { floor: 0, lowercase: true })} rollouts generated`
      : undefined;

  return [stepText, rolloutsText].filter(Boolean).join(' · ');
};

/** Epochs, and where the run is — the progress row beside the GRPO reward chart. */
export const getGrpoProgressTiles = (
  telemetry: CustomizationTrainingTelemetry,
  context: TrainingDiagnosticsContext
): StatTileDiagnosticProps[] => [
  {
    label: 'Epochs Completed',
    value:
      telemetry.epoch === undefined
        ? NOT_AVAILABLE
        : telemetry.numEpochs
          ? `${telemetry.epoch} / ${telemetry.numEpochs}`
          : String(telemetry.epoch),
  },
  getRunStateTile(telemetry, context),
];

export interface GrpoRunConfig {
  environment?: string;
  promptDataset?: string;
  trainingBackend: string;
  parallelism: string;
  generation: string;
  sequencePacking: string;
}

/** Joins config parts the way the run-configuration panel reads them. */
const joinParts = (...parts: (string | undefined)[]): string => parts.filter(Boolean).join(' · ');

/**
 * Spec values plus what `services/rl/.../nemo_rl/grpo_config.py` hardcodes. Anything that becomes
 * a real knob later turns into a spec read here, not a new row.
 */
export const getGrpoRunConfig = (
  spec: RlJobOutput & { training: RlGRPOTraining }
): GrpoRunConfig => {
  const { training } = spec;
  const parallelism = training.parallelism ?? {};
  const tensorParallel = parallelism.tensor_parallel_size ?? 1;
  const expertParallel = parallelism.expert_parallel_size ?? 1;

  // Megatron is inert for GRPO, so the policy is always DTensor; only v2 implements LoRA, expert
  // parallelism and `automodel_kwargs`, matching `_build_dtensor_cfg`.
  const isLora = training.finetuning_type === 'lora';
  // Keys, not truthiness: unset means "auto-detect", and `{}` is a truthy that would claim v2.
  const hasAutomodelKwargs = Object.keys(training.automodel_kwargs ?? {}).length > 0;
  const needsV2 = isLora || expertParallel > 1 || hasAutomodelKwargs;

  return {
    environment: spec.environment,
    promptDataset: spec.dataset,
    trainingBackend: joinParts(
      needsV2 ? 'DTensor v2' : 'DTensor',
      isLora ? 'LoRA' : 'Full weights'
    ),
    parallelism: joinParts(
      `TP ${tensorParallel}`,
      `PP ${parallelism.pipeline_parallel_size ?? 1}`,
      `CP ${parallelism.context_parallel_size ?? 1}`,
      expertParallel > 1 ? `EP ${expertParallel}` : undefined,
      parallelism.sequence_parallel ? 'sequence parallel' : undefined
    ),
    // Colocated with the policy and `async_grpo` off, so rollouts and training share the devices.
    generation: joinParts(
      'vLLM, colocated',
      `TP ${training.vllm_tensor_parallel_size ?? Math.min(tensorParallel, parallelism.num_gpus_per_node ?? tensorParallel)}`
    ),
    // Pinned off for GRPO by the service; it is a DPO-only knob today.
    sequencePacking: 'Disabled',
  };
};

const badge = (key: string, icon: ReactNode, label: string): ReactNode => (
  <Badge key={key} color="gray" kind="solid">
    {icon}
    {label}
  </Badge>
);

/**
 * Compute-configuration badges for a customization job. Automodel exposes distributed-training
 * `parallelism`; unsloth exposes single-node `hardware` (GPU list + precision).
 */
export const getTrainingOptionBadges = (job: CustomizationJob | null | undefined): ReactNode[] => {
  if (!job) return [];

  if (isAutomodelJob(job)) {
    const p = job.spec.parallelism;
    const badges: ReactNode[] = [
      badge('num_gpus_per_node', <Gpu />, getTextWithCount('GPU', p.num_gpus_per_node ?? 0)),
      badge('num_nodes', <Circle />, getTextWithCount('Node', p.num_nodes ?? 0)),
      badge(
        'tensor_parallel_size',
        <Gpu />,
        getTextWithCount('Tensor Parallel', p.tensor_parallel_size ?? 0)
      ),
    ];
    if (p.sequence_parallel) {
      badges.push(badge('sequence_parallel', undefined, 'Sequence Parallel'));
    }
    return badges;
  }

  if (isUnslothJob(job)) {
    const { gpus, precision } = job.spec.hardware ?? {};
    const badges: ReactNode[] = [];
    if (gpus) {
      const gpuCount = gpus.split(',').filter(Boolean).length;
      badges.push(badge('gpus', <Gpu />, getTextWithCount('GPU', gpuCount)));
    }
    badges.push(badge('precision', undefined, `Precision: ${precision}`));
    return badges;
  }

  return [];
};
