// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AutomodelJobOutput,
  AutomodelJobsJob,
  RlDPOTraining,
  RlGRPOTraining,
  RlJobOutput,
  RlJobsJob,
  UnslothJobOutput,
  UnslothJobsJob,
} from '@nemo/sdk/generated/customizer/schema';

export const CustomizationBackend = {
  automodel: 'automodel',
  unsloth: 'unsloth',
  rl: 'rl',
} as const;
export type CustomizationBackend = (typeof CustomizationBackend)[keyof typeof CustomizationBackend];

/** Narrows opaque JSON (a saved template's payload, location state) to a known backend. */
export const isCustomizationBackend = (value: unknown): value is CustomizationBackend =>
  Object.values(CustomizationBackend).includes(value as CustomizationBackend);

export type CustomizationJobSpec = AutomodelJobOutput | UnslothJobOutput | RlJobOutput;
export type AutomodelJob = AutomodelJobsJob;
export type UnslothJob = UnslothJobsJob;
export type RlJob = RlJobsJob;
export type CustomizationJob = AutomodelJobsJob | UnslothJobsJob | RlJobsJob;
export type CustomizationJobStatusDetails = Record<string, unknown>;

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

/**
 * Discriminator values for the RL training union. Orval inlines discriminator
 * literals rather than emitting a const object, so there is no generated enum to
 * import — `satisfies` pins these to the SDK types instead, failing the build if
 * the backend ever renames one.
 */
const RL_TRAINING_TYPES: readonly string[] = [
  'dpo' satisfies RlDPOTraining['type'],
  'grpo' satisfies RlGRPOTraining['type'],
];

export const isAutomodelSpec = (spec: unknown): spec is AutomodelJobOutput =>
  isObject(spec) && 'parallelism' in spec;

export const isUnslothSpec = (spec: unknown): spec is UnslothJobOutput =>
  isObject(spec) && 'hardware' in spec;

const hasRlTraining = (spec: unknown): boolean =>
  isObject(spec) &&
  !('parallelism' in spec) &&
  !('hardware' in spec) &&
  'training' in spec &&
  isObject((spec as { training: unknown }).training) &&
  RL_TRAINING_TYPES.includes((spec as { training: { type?: string } }).training.type ?? '');

export const isRlSpec = (spec: unknown): spec is RlJobOutput => hasRlTraining(spec);

export const getCustomizationBackend = (spec: unknown): CustomizationBackend | undefined => {
  if (isAutomodelSpec(spec)) return CustomizationBackend.automodel;
  if (isUnslothSpec(spec)) return CustomizationBackend.unsloth;
  if (isRlSpec(spec)) return CustomizationBackend.rl;
  return undefined;
};

export const isAutomodelJob = (job: CustomizationJob): job is AutomodelJob =>
  isAutomodelSpec(job.spec);

export const isUnslothJob = (job: CustomizationJob): job is UnslothJob => isUnslothSpec(job.spec);

export const isRlJob = (job: CustomizationJob): job is RlJob => isRlSpec(job.spec);

export const isDpoJob = (
  job: CustomizationJob
): job is RlJob & { spec: RlJobOutput & { training: RlDPOTraining } } =>
  isRlJob(job) && job.spec.training.type === 'dpo';

export const isGrpoJob = (
  job: CustomizationJob
): job is RlJob & { spec: RlJobOutput & { training: RlGRPOTraining } } =>
  isRlJob(job) && job.spec.training.type === 'grpo';
