// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RlGRPOTraining, RlJobOutput } from '@nemo/sdk/generated/customizer/schema';
import type { FinetuningType } from '@nemo/sdk/generated/platform/schema';
import type {
  CustomizationJob,
  CustomizationJobStatusDetails,
} from '@studio/util/customizationBackend';
import {
  getGrpoProgressTiles,
  getGrpoRunConfig,
  getGrpoRunProgressSummary,
  getGrpoSummaryTiles,
  formatFinetuningType,
  formatTrainingPhase,
  getBaseModel,
  getCustomizationTrainingProgress,
  getDatasetUri,
  getFinetuningType,
  getFormattedCustomizationStatus,
  getJobDuration,
  getJobStartDate,
  getFormattedTrainingType,
  formatMetricValue,
  formatStepCount,
  getLossTiles,
  getTrainingProgressTiles,
  getTrainingBatchSize,
  getTrainingDiagnosticsTiles,
  getTrainingTelemetry,
} from '@studio/util/customizations';

/** Minimal automodel job (carries `parallelism`). */
const automodelJob = (spec: Record<string, unknown>): CustomizationJob =>
  ({ spec: { parallelism: {}, ...spec } }) as unknown as CustomizationJob;

/** Minimal unsloth job (carries `hardware`). */
const unslothJob = (spec: Record<string, unknown>): CustomizationJob =>
  ({ spec: { hardware: {}, ...spec } }) as unknown as CustomizationJob;

/** Minimal RL job (no `parallelism`/`hardware` at the spec root, and a `training.type`). */
const rlJob = (spec: Partial<RlJobOutput>): CustomizationJob => ({
  name: 'rl-job',
  spec: {
    model: '',
    dataset: '',
    training: { type: 'grpo' },
    output: { name: '', fileset: '' },
    ...spec,
  },
});

describe('getFormattedTrainingType', () => {
  it('Returns the correctly formatted training type', () => {
    expect(getFormattedTrainingType('lora')).toEqual('LoRA');
    expect(getFormattedTrainingType('sft')).toEqual('SFT');
    expect(getFormattedTrainingType('distillation')).toEqual('Distillation');
    expect(getFormattedTrainingType('all_weights')).toEqual('All Weights');
    expect(getFormattedTrainingType('dpo')).toEqual('DPO');
    expect(getFormattedTrainingType('grpo')).toEqual('GRPO');
  });
});

describe('formatFinetuningType', () => {
  it.each([
    ['lora', 'LoRA'],
    ['lora_merged', 'LoRA Merged'],
    ['all_weights', 'All Weights'],
    ['dora', 'DoRA'],
  ])('formats %s as %s', (input, expected) => {
    expect(formatFinetuningType(input as FinetuningType)).toEqual(expected);
  });
});

describe('getFormattedCustomizationStatus', () => {
  it('Returns the correctly formatted status', () => {
    expect(getFormattedCustomizationStatus(undefined)).toEqual('');
    expect(getFormattedCustomizationStatus('created')).toEqual('Created');
    expect(getFormattedCustomizationStatus('completed')).toEqual('Completed');
    expect(getFormattedCustomizationStatus('running')).toEqual('Running');
  });

  it('appends progress percentage when provided', () => {
    expect(getFormattedCustomizationStatus('running', 45.7)).toEqual('Running (45%)');
  });
});

describe('getBaseModel', () => {
  it('returns empty string when job is undefined', () => {
    expect(getBaseModel(undefined)).toBe('');
  });

  it('returns model string for automodel jobs', () => {
    expect(getBaseModel(automodelJob({ model: 'my-model' }))).toBe('my-model');
  });

  it('returns model.name for unsloth jobs', () => {
    expect(getBaseModel(unslothJob({ model: { name: 'my-model' } }))).toBe('my-model');
  });

  it('returns empty string when the model is missing', () => {
    expect(getBaseModel(automodelJob({}))).toBe('');
  });

  it('returns the model reference for RL jobs, which used to read as blank', () => {
    expect(getBaseModel(rlJob({ model: 'qwen/qwen2.5-7b-instruct' }))).toBe(
      'qwen/qwen2.5-7b-instruct'
    );
  });
});

describe('getFinetuningType', () => {
  it('returns the finetuning type for GRPO, so the page header keeps that segment', () => {
    expect(getFinetuningType(rlJob({ training: { type: 'grpo', finetuning_type: 'lora' } }))).toBe(
      'lora'
    );
  });

  it('stays blank for DPO, which has no finetuning type', () => {
    expect(getFinetuningType(rlJob({ training: { type: 'dpo' } }))).toBe('');
  });
});

describe('getDatasetUri', () => {
  it('returns dataset.training for automodel jobs', () => {
    expect(getDatasetUri(automodelJob({ dataset: { training: 'urn:dataset:abc' } }))).toBe(
      'urn:dataset:abc'
    );
  });

  it('returns dataset.path for unsloth jobs', () => {
    expect(getDatasetUri(unslothJob({ dataset: { path: 'urn:dataset:xyz' } }))).toBe(
      'urn:dataset:xyz'
    );
  });

  it('returns empty string when job is undefined', () => {
    expect(getDatasetUri(undefined)).toBe('');
  });

  it('returns the bare fileset reference for RL jobs', () => {
    expect(getDatasetUri(rlJob({ dataset: 'default/deepscaler-train' }))).toBe(
      'default/deepscaler-train'
    );
  });
});

describe('getTrainingBatchSize', () => {
  it('uses global_batch_size for automodel', () => {
    expect(getTrainingBatchSize(automodelJob({ batch: { global_batch_size: 16 } }))).toBe(16);
  });

  it('uses per_device_train_batch_size for unsloth', () => {
    expect(getTrainingBatchSize(unslothJob({ batch: { per_device_train_batch_size: 4 } }))).toBe(4);
  });
});

describe('getCustomizationTrainingProgress', () => {
  it('returns empty string when no status_details', () => {
    const job = {} as unknown as CustomizationJob;
    expect(getCustomizationTrainingProgress(job)).toBe('');
  });

  it('returns empty string when epoch and percentage_done are both null', () => {
    const job = automodelJob({ schedule: { epochs: 5 } });
    (job as { status_details?: unknown }).status_details = {};
    expect(getCustomizationTrainingProgress(job)).toBe('');
  });

  it('returns formatted progress string', () => {
    const job = automodelJob({ schedule: { epochs: 5 } });
    (job as { status_details?: unknown }).status_details = { epoch: 2, percentage_done: 40 };
    expect(getCustomizationTrainingProgress(job)).toBe('2/5 (40%)');
  });

  it('handles missing epoch gracefully', () => {
    const job = automodelJob({ schedule: { epochs: 3 } });
    (job as { status_details?: unknown }).status_details = { percentage_done: 60 };
    expect(getCustomizationTrainingProgress(job)).toBe('0/3 (60%)');
  });
});

describe('getTrainingTelemetry', () => {
  const jobWithDetails = (details: Record<string, unknown>): CustomizationJob =>
    ({ status_details: details }) as unknown as CustomizationJob;

  it('returns an empty object when there is no status_details', () => {
    expect(getTrainingTelemetry(undefined)).toEqual({});
    expect(getTrainingTelemetry({} as CustomizationJob)).toEqual({});
  });

  it('coerces the live per-step telemetry keys', () => {
    expect(
      getTrainingTelemetry(
        jobWithDetails({
          phase: 'training',
          step: 4,
          max_steps: 10,
          num_epochs: 3,
          epoch: 1,
          train_loss: 0.42,
          val_loss: 0.55,
          train_lr: 0.000005,
          train_grad_norm: 1.25,
          checkpoint_path: 'ws/fileset/checkpoints/step-4',
        })
      )
    ).toEqual({
      phase: 'training',
      step: 4,
      maxSteps: 10,
      numEpochs: 3,
      epoch: 1,
      trainLoss: 0.42,
      valLoss: 0.55,
      learningRate: 0.000005,
      gradNorm: 1.25,
      checkpointPath: 'ws/fileset/checkpoints/step-4',
    });
  });

  it('reads the unqualified names jobs used before the phase prefix', () => {
    // Those jobs are already in the database and never change. Without the
    // fallback their Learning Rate and Gradient Norm render blank forever.
    expect(
      getTrainingTelemetry(
        jobWithDetails({
          train_loss: 0.42,
          lr: 0.000005,
          grad_norm: 1.25,
        })
      )
    ).toEqual({
      trainLoss: 0.42,
      learningRate: 0.000005,
      gradNorm: 1.25,
      phase: undefined,
      step: undefined,
      maxSteps: undefined,
      numEpochs: undefined,
      epoch: undefined,
      valLoss: undefined,
      checkpointPath: undefined,
    });
  });

  it('prefers the qualified name when a job carries both', () => {
    expect(
      getTrainingTelemetry(jobWithDetails({ train_lr: 0.000009, lr: 0.000005 })).learningRate
    ).toBe(0.000009);
  });

  it('drops non-finite numbers, empty strings, and wrong types', () => {
    expect(
      getTrainingTelemetry(
        jobWithDetails({
          phase: '',
          step: Number.NaN,
          train_lr: null,
          train_grad_norm: 'oops',
          checkpoint_path: '',
        })
      )
    ).toEqual({
      phase: undefined,
      step: undefined,
      maxSteps: undefined,
      numEpochs: undefined,
      epoch: undefined,
      trainLoss: undefined,
      valLoss: undefined,
      learningRate: undefined,
      gradNorm: undefined,
      checkpointPath: undefined,
    });
  });
});

describe('formatTrainingPhase', () => {
  it('maps known phases to friendly labels', () => {
    expect(formatTrainingPhase('checkpoint_saved')).toBe('Checkpoint Saved');
    expect(formatTrainingPhase('training')).toBe('Training');
  });

  it('title-cases unknown phases', () => {
    expect(formatTrainingPhase('some_new_phase')).toBe('Some New Phase');
  });
});

describe('getLossTiles', () => {
  it('dashes out both tiles when nothing has been reported', () => {
    expect(getLossTiles(undefined, true)).toEqual([
      { label: 'Final Training Loss', value: '—' },
      { label: 'Final Validation Loss', value: '—' },
    ]);
  });

  it('labels tiles "Latest" until the job reaches a terminal status', () => {
    const running = getLossTiles(undefined);
    expect(running[0].label).toBe('Latest Training Loss');
    expect(running[1].label).toBe('Latest Validation Loss');

    const finished = getLossTiles(undefined, true);
    expect(finished[0].label).toBe('Final Training Loss');
    expect(finished[1].label).toBe('Final Validation Loss');
  });

  it('builds a delta-from-start hint, tinted by improvement direction', () => {
    const tiles = getLossTiles(
      {
        metrics: {
          train_loss: [
            { step: 0, value: 1.5 },
            { step: 10, value: 1.04 },
          ],
          val_loss: [
            { step: 0, value: 1.2 },
            { step: 10, value: 1.3 },
          ],
        },
      },
      true
    );

    // Loss is lower-is-better, so a fall is "success" and a rise is "error".
    expect(tiles[0]).toEqual({
      label: 'Final Training Loss',
      value: '1.0400',
      hint: '-0.4600 from start',
      hintStatus: 'success',
    });
    expect(tiles[1]).toEqual({
      label: 'Final Validation Loss',
      value: '1.3000',
      hint: '+0.1000 from start',
      hintStatus: 'error',
    });
  });

  it('reads a flat metric as neutral rather than as a failure', () => {
    // A run that ended where it started has not gone wrong; it has gone nowhere. Tinting that red
    // reads as an error, which for a stalled reward is a real state, not a fault.
    const [training] = getLossTiles(
      {
        metrics: {
          train_loss: [
            { step: 0, value: 1.5 },
            { step: 10, value: 1.5 },
          ],
        },
      },
      true
    );

    expect(training).toEqual({
      label: 'Final Training Loss',
      value: '1.5000',
      hint: '+0.0000 from start',
      hintStatus: 'neutral',
    });
  });

  it('omits the delta hint when only one metric point has been reported', () => {
    const tiles = getLossTiles({ metrics: { train_loss: [{ step: 0, value: 1.5 }] } }, true);

    expect(tiles[0]).toEqual({ label: 'Final Training Loss', value: '1.5000' });
  });
});

describe('getGrpoSummaryTiles', () => {
  it('dashes out every tile when the run reported nothing', () => {
    expect(getGrpoSummaryTiles(undefined, true)).toEqual([
      { label: 'Final Mean Reward', value: '—', hint: 'across all sampled rollouts' },
      { label: 'Validation Reward', value: '—', hint: 'held-out prompts' },
      { label: 'Median Step Time', value: '—', hint: 'wall clock per step' },
      { label: 'Truncation Rate', value: '—', hint: 'hit the length limit' },
    ]);
  });

  it('tints a rising reward as success — the opposite of loss', () => {
    const [reward] = getGrpoSummaryTiles(
      {
        metrics: {
          train_reward: [
            { step: 1, value: 0.18 },
            { step: 500, value: 0.64 },
          ],
        },
      },
      true
    );

    // The delta sits beside the value rather than in the hint, because the tile keeps its context.
    expect(reward).toEqual({
      label: 'Final Mean Reward',
      value: '0.6400',
      trailingLabel: '+0.4600 from start',
      trailingLabelStatus: 'success',
      hint: 'across all sampled rollouts',
    });
  });

  it('names the step the last validation pass ran at', () => {
    const [, validation] = getGrpoSummaryTiles(
      { metrics: { val_accuracy: [{ step: 300, value: 0.58 }] } },
      true
    );

    expect(validation).toEqual({
      label: 'Validation Reward',
      value: '0.5800',
      hint: 'held-out prompts, step 300',
    });
  });

  it('summarises step time as the median of the run, not the latest step', () => {
    const [, , stepTime] = getGrpoSummaryTiles(
      {
        metrics: {
          'train_timing/total_step_time': [
            { step: 1, value: 29.8 },
            { step: 2, value: 33.6 },
            { step: 3, value: 34.8 },
            // A checkpoint save doubles one step; the median is what the run actually costs.
            { step: 4, value: 71.2 },
          ],
        },
      },
      true
    );

    expect(stepTime).toEqual({
      label: 'Median Step Time',
      value: '34.2s',
      hint: 'wall clock per step',
    });
  });

  it('renders the truncation rate as a percentage', () => {
    const [, , , truncation] = getGrpoSummaryTiles(
      { metrics: { train_truncation_rate: [{ step: 500, value: 0.041 }] } },
      true
    );

    expect(truncation).toEqual({
      label: 'Truncation Rate',
      value: '4.1%',
      hint: 'hit the length limit',
    });
  });

  it('labels the reward tile "Latest" until the job reaches a terminal status', () => {
    expect(getGrpoSummaryTiles(undefined)[0].label).toBe('Latest Mean Reward');
  });
});

describe('getGrpoRunProgressSummary', () => {
  const grpoJob = (
    statusDetails: CustomizationJobStatusDetails,
    training: Partial<RlGRPOTraining> = { num_generations_per_prompt: 8, num_prompts_per_step: 8 }
  ): CustomizationJob => {
    const job = rlJob({ training: { type: 'grpo', ...training } });
    return { ...job, status_details: statusDetails } as CustomizationJob;
  };

  it('reports the in-flight step and the rollouts it implies', () => {
    const job = grpoJob({ step: 340, max_steps: 500 });

    expect(getGrpoRunProgressSummary(job, getTrainingTelemetry(job), false)).toBe(
      'step 340 of 500 · 21.8k rollouts generated'
    );
  });

  it('keeps the "of" total once the job is terminal, so an early stop is visible', () => {
    const job = grpoJob({ step: 340, max_steps: 500 });

    expect(getGrpoRunProgressSummary(job, getTrainingTelemetry(job), true)).toBe(
      'step 340 of 500 · 21.8k rollouts generated'
    );
  });

  it('reports how many steps ran once the job is terminal with no known target', () => {
    const job = grpoJob({ step: 500 });

    expect(getGrpoRunProgressSummary(job, getTrainingTelemetry(job), true)).toBe(
      '500 steps ran · 32k rollouts generated'
    );
  });

  it('uses present tense for an in-flight job with no known target yet', () => {
    const job = grpoJob({ step: 12 });

    expect(getGrpoRunProgressSummary(job, getTrainingTelemetry(job), false)).toBe(
      'step 12 · 768 rollouts generated'
    );
  });

  it('omits the rollout count when the spec has no group-size/prompts-per-step to derive it from', () => {
    const job = grpoJob({ step: 10, max_steps: 100 }, {});

    expect(getGrpoRunProgressSummary(job, getTrainingTelemetry(job), false)).toBe('step 10 of 100');
  });

  it('falls back to batch_size when num_prompts_per_step is omitted, as the service derives it', () => {
    // The SDK documents num_prompts_per_step as derived from batch_size / num_generations_per_prompt
    // when unset, so their product is just batch_size — a real job commonly omits it.
    const job = grpoJob(
      { step: 10, max_steps: 100 },
      { num_generations_per_prompt: 8, batch_size: 64 }
    );

    expect(getGrpoRunProgressSummary(job, getTrainingTelemetry(job), false)).toBe(
      'step 10 of 100 · 640 rollouts generated'
    );
  });

  it('falls back to batch_size when num_generations_per_prompt is omitted, as the service derives it', () => {
    // The SDK documents num_generations_per_prompt as derived from batch_size / num_prompts_per_step
    // when unset, so their product is just batch_size — the mirror image of the case above.
    const job = grpoJob({ step: 10, max_steps: 100 }, { num_prompts_per_step: 8, batch_size: 64 });

    expect(getGrpoRunProgressSummary(job, getTrainingTelemetry(job), false)).toBe(
      'step 10 of 100 · 640 rollouts generated'
    );
  });

  it('is empty for a non-GRPO job', () => {
    const job = { ...rlJob({ training: { type: 'dpo' } }), status_details: { step: 10 } };

    expect(getGrpoRunProgressSummary(job, getTrainingTelemetry(job), false)).toBe('');
  });

  it('is empty when the job has not reported a step yet', () => {
    const job = grpoJob({});

    expect(getGrpoRunProgressSummary(job, getTrainingTelemetry(job), false)).toBe('');
  });
});

describe('getGrpoRunConfig', () => {
  const spec = (
    training: Partial<RlGRPOTraining> = {},
    rest: Record<string, unknown> = {}
  ): RlJobOutput & { training: RlGRPOTraining } =>
    ({
      model: 'qwen/qwen2.5-7b-instruct',
      dataset: 'default/deepscaler-train',
      environment: 'default/math-verifier-env',
      training: { type: 'grpo', ...training },
      ...rest,
    }) as RlJobOutput & { training: RlGRPOTraining };

  it('reads the environment and prompt dataset straight off the spec', () => {
    const config = getGrpoRunConfig(spec());

    expect(config.environment).toBe('default/math-verifier-env');
    expect(config.promptDataset).toBe('default/deepscaler-train');
  });

  it('reports DTensor, since Megatron is built inert for GRPO', () => {
    expect(getGrpoRunConfig(spec({ finetuning_type: 'all_weights' })).trainingBackend).toBe(
      'DTensor · Full weights'
    );
  });

  it.each([
    ['LoRA', { finetuning_type: 'lora' as const }],
    ['expert parallelism', { parallelism: { expert_parallel_size: 2 } }],
    ['automodel kwargs', { automodel_kwargs: { force_hf: true } }],
  ])('selects the v2 backend for %s, which only v2 implements', (_label, training) => {
    expect(getGrpoRunConfig(spec(training)).trainingBackend).toContain('DTensor v2');
  });

  it('does not read an empty automodel_kwargs as selecting v2', () => {
    // The schema says unset means "auto-detect", and `{}` is a plausible API round-trip of that.
    expect(getGrpoRunConfig(spec({ automodel_kwargs: {} })).trainingBackend).toBe(
      'DTensor · Full weights'
    );
  });

  it('spells out the parallelism plan, defaulting the unset degrees to 1', () => {
    expect(getGrpoRunConfig(spec({ parallelism: { tensor_parallel_size: 4 } })).parallelism).toBe(
      'TP 4 · PP 1 · CP 1'
    );

    expect(
      getGrpoRunConfig(
        spec({
          parallelism: {
            tensor_parallel_size: 4,
            context_parallel_size: 2,
            expert_parallel_size: 8,
            sequence_parallel: true,
          },
        })
      ).parallelism
    ).toBe('TP 4 · PP 1 · CP 2 · EP 8 · sequence parallel');
  });

  it('describes generation as colocated, which is what the service pins', () => {
    // Not the mockup's "async, separate GPUs": the service pins colocated with async off.
    expect(getGrpoRunConfig(spec({ vllm_tensor_parallel_size: 2 })).generation).toBe(
      'vLLM, colocated · TP 2'
    );
  });

  it('defaults the vLLM tensor parallel size to what fits in a node', () => {
    expect(
      getGrpoRunConfig(spec({ parallelism: { tensor_parallel_size: 8, num_gpus_per_node: 4 } }))
        .generation
    ).toBe('vLLM, colocated · TP 4');
  });

  it('reports sequence packing as disabled, which the service pins for GRPO', () => {
    expect(getGrpoRunConfig(spec()).sequencePacking).toBe('Disabled');
  });
});

describe('getGrpoProgressTiles', () => {
  it('adds the run-state tile to the epoch progress; step is on the page header instead', () => {
    expect(
      getGrpoProgressTiles(
        { step: 500, maxSteps: 500, epoch: 1, numEpochs: 1 },
        {
          isTerminal: true,
          duration: '00:42:11',
        }
      )
    ).toEqual([
      { label: 'Epochs Completed', value: '1 / 1' },
      { label: 'Duration', value: '00:42:11', hint: 'total run time' },
    ]);
  });

  it('shows the current phase while the run is still going', () => {
    const tiles = getGrpoProgressTiles(
      { phase: 'training' },
      {
        isTerminal: false,
        duration: '00:00:10',
      }
    );

    expect(tiles[1]).toEqual({ label: 'Phase', value: 'Training', hint: 'current stage' });
  });

  it('names the failing step instead of the duration when the run errored', () => {
    const tiles = getGrpoProgressTiles(
      { phase: 'training', epoch: 0, numEpochs: 1 },
      {
        isTerminal: true,
        duration: '00:48:38',
        failedAtStepLabel: 'GRPO training',
      }
    );

    expect(tiles[1]).toEqual({
      label: 'Run State',
      value: 'Failed',
      hint: 'during GRPO training',
      hintStatus: 'error',
      status: 'error',
    });
  });

  it('marks the failure on the hint, which is the only part styled on an unbordered tile', () => {
    const [, runState] = getGrpoProgressTiles(
      {},
      { isTerminal: true, duration: '48m 38s', failedAtStepLabel: 'model upload' }
    );

    expect(runState.hintStatus).toBe('error');
  });
});

describe('getTrainingProgressTiles', () => {
  it('dashes out both tiles when telemetry is empty', () => {
    expect(getTrainingProgressTiles({})).toEqual([
      { label: 'Steps Completed', value: '—' },
      { label: 'Epochs Completed', value: '—' },
    ]);
  });

  it('renders step and epoch progress against their totals', () => {
    expect(getTrainingProgressTiles({ step: 40, maxSteps: 100, epoch: 1, numEpochs: 3 })).toEqual([
      { label: 'Steps Completed', value: '40 / 100' },
      { label: 'Epochs Completed', value: '1 / 3' },
    ]);
  });

  it('drops the denominator when the total is unknown', () => {
    expect(getTrainingProgressTiles({ step: 40, epoch: 1 })).toEqual([
      { label: 'Steps Completed', value: '40' },
      { label: 'Epochs Completed', value: '1' },
    ]);
  });

  it('keeps a zeroth step rather than treating it as missing', () => {
    expect(getTrainingProgressTiles({ step: 0, maxSteps: 100 })[0]).toEqual({
      label: 'Steps Completed',
      value: '0 / 100',
    });
  });
});

describe('getJobDuration', () => {
  const step = (created_at: string, updated_at: string) => ({ created_at, updated_at });

  const completedSteps = [
    step('2026-08-14T13:21:46.320423', '2026-08-14T13:22:04.380207'),
    step('2026-08-14T13:22:04.395860', '2026-08-14T13:24:46.508047'),
    step('2026-08-14T13:24:46.524450', '2026-08-14T13:25:00.440349'),
    step('2026-08-14T13:25:00.500539', '2026-08-14T13:25:06.492130'),
  ];

  it('ticks live off the elapsed seconds while the job is running', () => {
    expect(getJobDuration(completedSteps, false, 72)).toBe('1m 12s');
  });

  it('spans the first step start to the last step end once terminal', () => {
    expect(getJobDuration(completedSteps, true)).toBe('00:03:20');
  });

  it('parses timestamps as UTC even when only one carries a timezone', () => {
    expect(getJobDuration([step('2025-06-25T21:41:02Z', '2025-06-25T21:42:14.242833')], true)).toBe(
      '00:01:12'
    );
  });

  it('reports a dash rather than 00:00:00 when a step never advanced', () => {
    expect(getJobDuration([step('2025-06-25T21:41:02', '2025-06-25T21:41:02')], true)).toBe('—');
  });

  it('reports a dash when there are no steps or nothing has elapsed yet', () => {
    expect(getJobDuration([], true)).toBe('—');
    expect(getJobDuration(undefined, true)).toBe('—');
    expect(getJobDuration(completedSteps, false, 0)).toBe('—');
    expect(getJobDuration(completedSteps, false, undefined)).toBe('—');
  });
});

describe('getJobStartDate', () => {
  it('anchors to the first step, not the job record', () => {
    const start = getJobStartDate([
      { created_at: '2026-08-14T13:21:46.320423', updated_at: '2026-08-14T13:22:04.380207' },
      { created_at: '2026-08-14T13:22:04.395860', updated_at: '2026-08-14T13:24:46.508047' },
    ]);

    expect(start?.toISOString()).toBe('2026-08-14T13:21:46.320Z');
  });

  it('returns undefined without steps', () => {
    expect(getJobStartDate(undefined)).toBeUndefined();
    expect(getJobStartDate([])).toBeUndefined();
  });
});

describe('getTrainingDiagnosticsTiles', () => {
  const running = { isTerminal: false, duration: '1m 12s' };
  const finished = { isTerminal: true, duration: '00:01:12' };

  it('dashes out every value when nothing has been reported, keeping the grid stable', () => {
    expect(
      getTrainingDiagnosticsTiles({}, undefined, running).map((tile) => [tile.label, tile.value])
    ).toEqual([
      ['Learning Rate', '—'],
      ['Gradient Norm', '—'],
      ['Train/Val Gap', '—'],
      ['Phase', '—'],
    ]);
  });

  it('dashes out only the values that are missing', () => {
    const tiles = getTrainingDiagnosticsTiles({ gradNorm: 1.5 }, undefined, running);

    expect(tiles).toHaveLength(4);
    expect(tiles.map((tile) => [tile.label, tile.value])).toEqual([
      ['Learning Rate', '—'],
      ['Gradient Norm', '1.5000'],
      ['Train/Val Gap', '—'],
      ['Phase', '—'],
    ]);
  });

  it('keeps a reported zero rather than mistaking it for a missing value', () => {
    const tiles = getTrainingDiagnosticsTiles({ learningRate: 0, gradNorm: 0 }, undefined, running);

    expect(tiles.slice(0, 2).map((tile) => tile.value)).toEqual(['0.00e+0', '0.0000']);
  });

  it('shows the current phase while the job is still running', () => {
    const tiles = getTrainingDiagnosticsTiles(
      { learningRate: 0.000005, gradNorm: 1.2345, phase: 'checkpoint_saved' },
      undefined,
      running
    );

    expect(tiles.map((tile) => [tile.label, tile.value])).toEqual([
      ['Learning Rate', '5.00e-6'],
      ['Gradient Norm', '1.2345'],
      ['Train/Val Gap', '—'],
      ['Phase', 'Checkpoint Saved'],
    ]);
  });

  it('swaps phase for duration once terminal, since phase then repeats the status badge', () => {
    const tiles = getTrainingDiagnosticsTiles(
      { learningRate: 0.000005, gradNorm: 1.2345, phase: 'completed' },
      undefined,
      finished
    );

    expect(tiles.map((tile) => [tile.label, tile.value])).toEqual([
      ['Learning Rate', '5.00e-6'],
      ['Gradient Norm', '1.2345'],
      ['Train/Val Gap', '—'],
      ['Duration', '00:01:12'],
    ]);
  });

  it('derives the generalization gap from the final points of both loss series', () => {
    const tiles = getTrainingDiagnosticsTiles(
      {},
      {
        metrics: {
          train_loss: [
            { step: 0, value: 5.14 },
            { step: 10, value: 1.181 },
          ],
          val_loss: [
            { step: 0, value: 1.86 },
            { step: 10, value: 1.84 },
          ],
        },
      },
      running
    );

    expect(tiles[2]).toEqual({
      label: 'Train/Val Gap',
      value: '0.6590',
      hint: 'validation - training',
    });
  });

  it('reports a negative gap when validation is ahead of training', () => {
    const tiles = getTrainingDiagnosticsTiles(
      {},
      {
        metrics: {
          train_loss: [{ step: 0, value: 1.5 }],
          val_loss: [{ step: 0, value: 1.2 }],
        },
      },
      running
    );

    expect(tiles[2]).toMatchObject({ label: 'Train/Val Gap', value: '-0.3000' });
  });

  it('dashes the gap when only one loss series is present', () => {
    const tiles = getTrainingDiagnosticsTiles(
      {},
      { metrics: { train_loss: [{ step: 0, value: 1.5 }] } },
      running
    );

    expect(tiles[2]).toMatchObject({ label: 'Train/Val Gap', value: '—' });
  });
});

describe('getFinetuningType', () => {
  it('returns an empty string when the job or training block is missing', () => {
    expect(getFinetuningType(undefined)).toBe('');
    expect(getFinetuningType(automodelJob({}))).toBe('');
    expect(getFinetuningType(unslothJob({}))).toBe('');
  });

  it('reads the finetuning type from either backend', () => {
    expect(getFinetuningType(automodelJob({ training: { finetuning_type: 'lora' } }))).toBe('lora');
    expect(getFinetuningType(unslothJob({ training: { finetuning_type: 'all_weights' } }))).toBe(
      'all_weights'
    );
  });
});

describe('formatMetricValue', () => {
  it('uses fixed notation for values a training run normally produces', () => {
    expect(formatMetricValue(0.6964)).toBe('0.6964');
    expect(formatMetricValue(-0.46)).toBe('-0.4600');
    expect(formatMetricValue(0)).toBe('0.0000');
  });

  it('switches to exponential past the magnitude that would distort the tile grid', () => {
    // toFixed(4) here would be "12000000000.0000" — 16 characters.
    expect(formatMetricValue(1.2e10)).toBe('1.20e+10');
    expect(formatMetricValue(-1.2e10)).toBe('-1.20e+10');
  });

  it('keeps fixed notation right below the threshold', () => {
    expect(formatMetricValue(999_999)).toBe('999999.0000');
    expect(formatMetricValue(1e6)).toBe('1.00e+6');
  });
});

describe('formatStepCount', () => {
  it('leaves ordinary step counts alone', () => {
    expect(formatStepCount(0)).toBe('0');
    expect(formatStepCount(210)).toBe('210');
    expect(formatStepCount(9_999)).toBe('9,999');
  });

  it('compacts counts long enough to widen the column', () => {
    expect(formatStepCount(1_234_567)).toBe('1.2M');
    expect(formatStepCount(10_000)).toBe('10K');
  });

  it('lowers the floor and the suffix case when asked, for prose rather than a tile', () => {
    expect(formatStepCount(210, { floor: 0 })).toBe('210');
    expect(formatStepCount(174_000, { floor: 0, lowercase: true })).toBe('174k');
    expect(formatStepCount(9_999, { floor: 0, lowercase: true })).toBe('10k');
  });
});
