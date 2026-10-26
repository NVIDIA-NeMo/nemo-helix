// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from '@storybook/react';
import { TrainingLossPanel } from '@studio/components/CustomizationOverview/TrainingLossPanel';
import {
  getCurrentStage,
  getTrainingDiagnosticsTiles,
  getTrainingProgressTiles,
  type JobStageStep,
} from '@studio/util/customizations';

const STAGE_STEPS = {
  download: 'model-and-dataset-download',
  training: 'training',
  upload: 'model-upload',
  register: 'model-entity-creation',
} as const;
const STAGE_ORDER = Object.values(STAGE_STEPS);

const PHASES = [
  'none',
  'downloading',
  'conversions',
  'compiling_config',
  'training',
  'validation',
  'saving_checkpoint',
  'checkpoint_saved',
  'epoch_end',
  'merging_adapter',
  'finalizing_checkpoint',
  'uploading',
  'registering_model',
] as const;

interface PlaygroundArgs {
  /** `between-steps` is every created step completed, before the next one exists. */
  stage: keyof typeof STAGE_STEPS | 'between-steps';
  stepStatus: 'created' | 'pending' | 'active' | 'resuming' | 'pausing' | 'paused' | 'cancelling';
  /** What the step's task reports as `phase`. */
  phase: (typeof PHASES)[number];
  /** Where the job clock reads, relative to the start of the current stage. */
  secondsIntoStage: number;
  step: number;
  maxSteps: number;
  /** A retried step holds a dead pod's task beside the live one. */
  staleErroredTask: boolean;
  /** Reported with the `uploading` phase; 0 total leaves them out, as before the upload starts. */
  uploadedGiB: number;
  totalGiB: number;
}

const STEP_GAP_SECONDS = 120;

const GIB = 1024 ** 3;

const buildSteps = ({
  stage,
  stepStatus,
  phase,
  staleErroredTask,
  uploadedGiB,
  totalGiB,
}: PlaygroundArgs) => {
  const currentIndex =
    stage === 'between-steps' ? STAGE_ORDER.length : STAGE_ORDER.indexOf(STAGE_STEPS[stage]);
  const created = Math.min(currentIndex + 1, STAGE_ORDER.length);
  const steps: JobStageStep[] = STAGE_ORDER.slice(0, created).map((name, index) => {
    const isCurrent = index === currentIndex;
    const status = isCurrent ? stepStatus : 'completed';
    return {
      name,
      status,
      created_at: new Date(
        Date.UTC(2026, 9, 14, 10, 0, 0) + index * STEP_GAP_SECONDS * 1000
      ).toISOString(),
      tasks: [
        ...(isCurrent && staleErroredTask
          ? [
              {
                status: 'error',
                updated_at: '2026-10-14T09:00:00Z',
                status_details: { phase: 'training' },
              },
            ]
          : []),
        {
          status: isCurrent ? 'active' : 'completed',
          updated_at: '2026-10-14T11:00:00Z',
          status_details: isCurrent
            ? phase === 'none'
              ? {}
              : {
                  phase,
                  // The upload task counts bytes in flight, so they move while one file streams.
                  ...(phase === 'uploading' && totalGiB > 0
                    ? { uploaded_bytes: uploadedGiB * GIB, total_size: totalGiB * GIB }
                    : {}),
                }
            : { phase: 'completed' },
        },
      ],
    };
  });
  return steps;
};

const meta: Meta<PlaygroundArgs> = {
  title: 'Customization/Training stage',
  args: {
    stage: 'training',
    stepStatus: 'active',
    phase: 'saving_checkpoint',
    secondsIntoStage: 160,
    step: 125,
    maxSteps: 125,
    staleErroredTask: false,
    uploadedGiB: 0,
    totalGiB: 0,
  },
  argTypes: {
    stage: {
      control: 'select',
      options: [...Object.keys(STAGE_STEPS), 'between-steps'],
    },
    stepStatus: {
      control: 'select',
      options: ['created', 'pending', 'active', 'resuming', 'pausing', 'paused', 'cancelling'],
    },
    phase: { control: 'select', options: PHASES },
    secondsIntoStage: { control: { type: 'number', min: 0, step: 10 } },
    step: { control: { type: 'number', min: 0 } },
    maxSteps: { control: { type: 'number', min: 0 } },
    staleErroredTask: { control: 'boolean' },
    uploadedGiB: { control: { type: 'number', min: 0, step: 0.5 } },
    totalGiB: { control: { type: 'number', min: 0, step: 1 } },
  },
  // The page's own composition for an active job, with only the steps varying.
  render: (args) => {
    const steps = buildSteps(args);
    const jobSeconds = (steps.length - 1) * STEP_GAP_SECONDS + args.secondsIntoStage;
    const telemetry = { step: args.step, maxSteps: args.maxSteps, epoch: 1, numEpochs: 1 };
    return (
      <TrainingLossPanel
        jobStatus="active"
        maxSteps={args.maxSteps}
        metrics={getTrainingDiagnosticsTiles(telemetry, undefined, {
          isTerminal: false,
          duration: '00:42:10',
          currentStage: getCurrentStage(steps, jobSeconds),
        })}
        progress={getTrainingProgressTiles(telemetry)}
      />
    );
  },
};

export default meta;

type Story = StoryObj<PlaygroundArgs>;

/** Change anything in the Controls panel; the tile is built by the same code as the page. */
export const Playground: Story = {};

export const Downloading: Story = {
  args: { stage: 'download', phase: 'downloading', step: 0, maxSteps: 0 },
};
export const CompilingConfig: Story = { args: { phase: 'compiling_config', step: 0 } };
export const Training: Story = { args: { phase: 'training', step: 60 } };
export const SavingCheckpoint: Story = {};
export const MergingAdapter: Story = { args: { phase: 'merging_adapter' } };
export const Paused: Story = { args: { stepStatus: 'paused', phase: 'training', step: 60 } };
export const BetweenSteps: Story = { args: { stage: 'between-steps' } };
export const Uploading: Story = {
  args: { stage: 'upload', phase: 'uploading', uploadedGiB: 12.6, totalGiB: 63 },
};
export const Registering: Story = { args: { stage: 'register', phase: 'registering_model' } };
