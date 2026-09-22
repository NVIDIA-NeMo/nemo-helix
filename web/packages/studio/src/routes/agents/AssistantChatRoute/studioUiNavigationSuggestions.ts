// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { featureFlags } from '@studio/constants/featureFlags';
import type { FeatureFlags } from '@studio/constants/featureFlags/featureFlags';
import {
  getAgentsListRoute,
  getDataDesignerJobListRoute,
  getEvaluationResultsRoute,
  getGuardrailsRoute,
  getIntakeTracesRoute,
  getModelCompareRoute,
  getNewDataDesignerJobRoute,
  getNewFilesetRoute,
  getNewSafeSynthesizerRoute,
  getSecretsRoute,
  getWorkspaceBaseModelsRoute,
  getWorkspaceDeploymentsRoute,
  getWorkspaceInferenceProvidersRoute,
  getWorkspaceJobsRoute,
  getWorkspaceMembersRoute,
  getWorkspaceSafeSynthesizerRoute,
  getWorkspaceSettingsRoute,
} from '@studio/routes/utils';

type FeatureFlagName = keyof FeatureFlags;

export interface StudioUiNavigationSuggestion {
  id: string;
  title: string;
  description: string;
  href: string;
}

interface StudioUiDestination {
  id: string;
  title: string;
  description: string;
  getHref: (workspace: string) => string;
  patterns: readonly RegExp[];
  requiredFeatureFlags?: readonly FeatureFlagName[];
}

const STUDIO_UI_DESTINATIONS: readonly StudioUiDestination[] = [
  {
    id: 'safe-synthesizer-new',
    title: 'Open Safe Synthesizer',
    description: 'Studio has a guided UI for generating safe synthetic datasets.',
    getHref: getNewSafeSynthesizerRoute,
    requiredFeatureFlags: ['safeSynthesizerEnabled'],
    patterns: [
      /\bsafe[-\s]?synthesizer\b/i,
      /\bsynthetic (data|dataset|datasets)\b/i,
      /\bsynthesi[sz]e (data|dataset|datasets)\b/i,
      /\bgenerate (safety[-\s]?focused|safe|synthetic) (data|dataset|datasets)\b/i,
      /\bsafety data\b/i,
    ],
  },
  {
    id: 'guardrails',
    title: 'Open Guardrails',
    description: 'Studio has a UI for managing NeMo Guardrails configurations.',
    getHref: getGuardrailsRoute,
    requiredFeatureFlags: ['guardrailsEnabled'],
    patterns: [
      /\bguardrails?\b/i,
      /\bcontent safety\b/i,
      /\bjailbreak\b/i,
      /\bpii (redaction|guard|protection)\b/i,
      /\bguardrail middleware\b/i,
    ],
  },
  {
    id: 'data-designer-new',
    title: 'Open Data Designer',
    description: 'Studio has a Data Designer UI for creating and transforming datasets.',
    getHref: getNewDataDesignerJobRoute,
    requiredFeatureFlags: ['dataDesignerEnabled'],
    patterns: [
      /\bdata designer\b/i,
      /\bgenerate (synthetic )?(data|dataset|datasets)\b/i,
      /\bcreate (a )?(dataset|datasets)\b/i,
      /\btransform (a )?(dataset|datasets)\b/i,
      /\bdata generation (workflow|pipeline)\b/i,
    ],
  },
  {
    id: 'fileset-new',
    title: 'Open Fileset Upload',
    description: 'Studio has a UI for creating filesets and uploading files.',
    getHref: getNewFilesetRoute,
    requiredFeatureFlags: ['datasetsEnabled'],
    patterns: [
      /\bcreate (a )?fileset\b/i,
      /\bnew fileset\b/i,
      /\bupload (a )?(file|files|dataset|datasets)\b/i,
      /\bimport (a )?(file|files|dataset|datasets)\b/i,
    ],
  },
  {
    id: 'inference-providers',
    title: 'Open Inference Providers',
    description: 'Studio has a UI for adding and managing inference providers.',
    getHref: getWorkspaceInferenceProvidersRoute,
    requiredFeatureFlags: ['inferenceProviderEnabled'],
    patterns: [
      /\binference provider\b/i,
      /\bmodel provider\b/i,
      /\bconfigure inference (for|in) (this )?workspace\b/i,
      /\bconfigure (a )?ne?mo inference\b/i,
      /\b(add|create|configure|connect|manage|register) (an? )?(provider|inference endpoint)\b/i,
      /\b(connect|configure) (openai|nvidia|nim|build)\b/i,
    ],
  },
  {
    id: 'secrets',
    title: 'Open Secrets',
    description: 'Studio has a UI for creating and managing workspace secrets.',
    getHref: getSecretsRoute,
    requiredFeatureFlags: ['secretsEnabled'],
    patterns: [
      /\b(add|create|manage|store|update) (an? )?(workspace )?(secret|secrets)\b/i,
      /\b(add|create|manage|store|update) (an? )?(api key|credential|credentials|token) (secret|secrets|in (the )?workspace|for (this )?workspace)\b/i,
      /\bworkspace (api key|credential|credentials|token|secret|secrets)\b/i,
    ],
  },
  {
    id: 'model-playground',
    title: 'Open Playground',
    description: 'Studio has a playground UI for chatting with and comparing models.',
    getHref: getModelCompareRoute,
    requiredFeatureFlags: ['modelCompareEnabled'],
    patterns: [
      /\bmodel playground\b/i,
      /\bopen (the )?playground (for|with) (a )?model\b/i,
      /\bcompare (models|model responses)\b/i,
      /\bchat with (a )?model\b/i,
    ],
  },
  {
    id: 'model-deployments',
    title: 'Open Deployments',
    description: 'Studio has a UI for managing model deployments.',
    getHref: getWorkspaceDeploymentsRoute,
    requiredFeatureFlags: ['deploymentsEnabled'],
    patterns: [
      /\bmodel deployments?\b/i,
      /\bdeploy (a )?model\b/i,
      /\bserve (a )?model\b/i,
      /\bmodel endpoint\b/i,
    ],
  },
  {
    id: 'base-models',
    title: 'Open Base Models',
    description: 'Studio has a UI for browsing base models and model details.',
    getHref: getWorkspaceBaseModelsRoute,
    requiredFeatureFlags: ['baseModelsEnabled'],
    patterns: [
      /\bbase models?\b/i,
      /\bmodel catalog\b/i,
      /\bbrowse models?\b/i,
      /\blist models?\b/i,
    ],
  },
  {
    id: 'agents',
    title: 'Open Agents',
    description: 'Studio has a UI for viewing agents and managing their deployments.',
    getHref: getAgentsListRoute,
    requiredFeatureFlags: ['agentsEnabled'],
    patterns: [
      /\bmanage agents?\b/i,
      /\bview agents?\b/i,
      /\b(build|create) (an? )?agent\b(?!\s+(class|component|helper|test|function|module))\b/i,
      /\bcreate example agent\b/i,
      /\bclone (an? )?agent\b/i,
      /\bchat with (an? )?agent\b/i,
      /\bdeploy (an? )?agent\b/i,
      /\btry (a )?deployed agent\b/i,
    ],
  },
  {
    id: 'evaluations',
    title: 'Open Evaluations',
    description: 'Studio has a UI for reviewing model evaluation results.',
    getHref: getEvaluationResultsRoute,
    requiredFeatureFlags: ['evaluatorEnabled'],
    patterns: [
      /\bevaluat(e|ing|ion)s? (a )?model\b/i,
      /\bmodel (eval|evaluation|evaluations)\b/i,
      /\bevaluation results?\b/i,
      /\b(eval|evaluation) history\b/i,
      /\bnemo[-\s]?evaluator\b/i,
      /\bevaluator (jobs?|sdk specs?)\b/i,
      /\buse (the )?evaluator plugin\b/i,
    ],
  },
  {
    id: 'safe-synthesizer',
    title: 'Open Safe Synthesizer',
    description: 'Studio has a UI for monitoring safe synthetic data jobs.',
    getHref: getWorkspaceSafeSynthesizerRoute,
    requiredFeatureFlags: ['safeSynthesizerEnabled'],
    patterns: [/\bsafe synth(esizer)? jobs?\b/i],
  },
  {
    id: 'data-designer',
    title: 'Open Data Designer',
    description: 'Studio has a UI for managing Data Designer jobs.',
    getHref: getDataDesignerJobListRoute,
    requiredFeatureFlags: ['dataDesignerEnabled'],
    patterns: [/\bdata designer jobs?\b/i],
  },
  {
    id: 'jobs',
    title: 'Open Jobs',
    description: 'Studio has a UI for viewing workspace jobs.',
    getHref: getWorkspaceJobsRoute,
    requiredFeatureFlags: ['jobsEnabled'],
    patterns: [/\bworkspace jobs?\b/i, /\bworkspace job history\b/i],
  },
  {
    id: 'traces',
    title: 'Open Traces',
    description: 'Studio has a UI for inspecting intake traces and recording feedback.',
    getHref: getIntakeTracesRoute,
    requiredFeatureFlags: ['intakeEnabled'],
    patterns: [
      /\bintake (annotation|annotations|traces?|trace review)\b/i,
      /\b(annotation|annotations) (for|in|on) (intake|trace|traces)\b/i,
      /\b(trace|traces).*\b(annotation|annotations|review)\b/i,
    ],
  },
  {
    id: 'members',
    title: 'Open Members',
    description: 'Studio has a UI for managing workspace members.',
    getHref: getWorkspaceMembersRoute,
    requiredFeatureFlags: ['membersEnabled'],
    patterns: [/\bworkspace members?\b/i, /\badd (a )?member\b/i, /\buser access\b/i],
  },
  {
    id: 'settings',
    title: 'Open Settings',
    description: 'Studio has a UI for workspace settings.',
    getHref: getWorkspaceSettingsRoute,
    requiredFeatureFlags: ['settingsEnabled'],
    patterns: [
      /\bworkspace settings?\b/i,
      /\b(open|change|manage|update) (the )?settings (for|in|of) (this )?workspace\b/i,
    ],
  },
];

const isDestinationEnabled = (destination: StudioUiDestination): boolean =>
  destination.requiredFeatureFlags?.every((flag) => featureFlags[flag] !== false) ?? true;

export const getStudioUiNavigationSuggestion = (
  prompt: string,
  workspace: string
): StudioUiNavigationSuggestion | undefined => {
  const trimmedPrompt = prompt.trim();
  if (!trimmedPrompt) return undefined;

  for (const destination of STUDIO_UI_DESTINATIONS) {
    if (!isDestinationEnabled(destination)) continue;
    if (!destination.patterns.some((pattern) => pattern.test(trimmedPrompt))) continue;

    return {
      id: destination.id,
      title: destination.title,
      description: destination.description,
      href: destination.getHref(workspace),
    };
  }

  return undefined;
};
