// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  isYamlPath,
  looksLikeOptimizeConfig,
  optimizeBundleProblems,
  parseOptimizeConfig,
} from '@studio/api/agents/optimizeBundle';
import type { BundleFileSpec } from '@studio/components/BundleSourcePicker/types';

export type OptimizeConfig = Record<string, unknown>;

/** An optimize bundle's driving file, preflighted as `nemo agents optimize prepare-fileset` would. */
export const optimizeConfigSpec = (agentName: string): BundleFileSpec<OptimizeConfig> => ({
  label: 'optimize config',
  description: 'a YAML file with an optimizer section',
  candidateNoun: 'YAML files',
  isCandidate: isYamlPath,
  parse: (text) => {
    const config = parseOptimizeConfig(text);
    return config && looksLikeOptimizeConfig(config) ? config : undefined;
  },
  validate: (config, bundlePaths) =>
    optimizeBundleProblems({ config, bundlePaths, agent: agentName }),
});
