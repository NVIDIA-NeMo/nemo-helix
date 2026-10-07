// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import type { CreateAgentRequestConfig } from '@nemo/sdk/generated/agents/schema/CreateAgentRequestConfig';
import type { FilesetLocation } from '@studio/api/files/types';
import type { UseMutationOptions } from '@tanstack/react-query';

export interface CreateAgentFromTrialParams {
  workspace: string;
  name: string;
  /** The agent the study optimized, whose spec fileset the new agent needs a copy of. */
  source: FilesetLocation & Pick<Agent, 'description' | 'config_format'>;
  config: CreateAgentRequestConfig;
  replaceOrphanedFileset?: boolean;
}

export type UseCreateAgentFromTrialOptions = Omit<
  UseMutationOptions<Agent, Error, CreateAgentFromTrialParams>,
  'mutationFn'
>;
