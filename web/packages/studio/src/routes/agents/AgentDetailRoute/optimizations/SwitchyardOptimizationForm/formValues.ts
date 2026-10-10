// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { nameSchema } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import { z } from 'zod';

/** The routing strategies the `switchyard` strategy can build a VirtualModel for. */
export type RoutingStrategyId = 'random_routing' | 'stage_router' | 'llm_classifier';

/** The spec field each strategy's one tuning knob is submitted as. */
export type RoutingThresholdField =
  | 'strong_probability'
  | 'confidence_threshold'
  | 'base_threshold';

export interface RoutingStrategyOption {
  id: RoutingStrategyId;
  title: string;
  description: string;
  threshold: {
    field: RoutingThresholdField;
    label: string;
    help: string;
  };
}

/** Mirrors `SwitchyardOptimizeSpec` in the nemo-switchyard plugin: one knob per strategy, all 0–1. */
export const ROUTING_STRATEGIES: RoutingStrategyOption[] = [
  {
    id: 'random_routing',
    title: 'Random routing',
    description: 'Splits traffic between the pair at a fixed rate. A baseline to beat.',
    threshold: {
      field: 'strong_probability',
      label: 'Capable model share',
      help: 'Share of requests sent to the capable model.',
    },
  },
  {
    id: 'stage_router',
    title: 'Stage router',
    description:
      'Tries the efficient model first and escalates to the capable one when it is unsure.',
    threshold: {
      field: 'confidence_threshold',
      label: 'Confidence threshold',
      help: 'Confidence below which a request escalates to the capable model.',
    },
  },
  {
    id: 'llm_classifier',
    title: 'LLM classifier',
    description: 'A judge model scores each request and sends the hard ones to the capable model.',
    threshold: {
      field: 'base_threshold',
      label: 'Capability threshold',
      help: 'Capability score above which a request escalates to the capable model.',
    },
  },
];

const thresholdSchema = z.coerce
  .number({ invalid_type_error: 'Enter a number.' })
  .min(0, 'Must be between 0 and 1.')
  .max(1, 'Must be between 0 and 1.');

export const switchyardFormSchema = z.object({
  name: nameSchema,
  models: z
    .array(z.string().min(1))
    .min(2, 'Add at least two models to route between.')
    .refine((models) => new Set(models).size === models.length, 'Each model can be added once.'),
  routingStrategies: z
    .array(z.enum(['random_routing', 'stage_router', 'llm_classifier']))
    .min(1, 'Pick at least one routing strategy.'),
  strong_probability: thresholdSchema,
  confidence_threshold: thresholdSchema,
  base_threshold: thresholdSchema,
  /** Empty means each pair's capable model judges, which is the plugin's own default. */
  judgeModel: z.string(),
});

export type SwitchyardFormValues = {
  name: string;
  /** Model entity refs, most capable first. */
  models: string[];
  routingStrategies: RoutingStrategyId[];
  strong_probability: number;
  confidence_threshold: number;
  base_threshold: number;
  judgeModel: string;
};

export type SwitchyardFormOutput = z.output<typeof switchyardFormSchema>;

export interface ModelPair {
  capable: string;
  efficient: string;
}

/**
 * Every pair the plugin routes, in its order (`itertools.combinations`): each model is the capable
 * side against every model listed after it.
 */
export const modelPairs = (models: string[]): ModelPair[] =>
  models.flatMap((capable, index) =>
    models.slice(index + 1).map((efficient) => ({ capable, efficient }))
  );
